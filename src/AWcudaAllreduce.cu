#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>

/* TODO : cuda Stream to overlap compression and quant prediction*/

#define MPI_call_check(call)                                                   \
  {                                                                            \
    int err_code = call;                                                       \
    if (err_code != MPI_SUCCESS) {                                             \
      char error_string[BUFSIZ];                                               \
      int length_of_error_string;                                              \
      MPI_Error_string(err_code, error_string, &length_of_error_string);       \
      fprintf(stderr, "\nMPI error in line %d : %s\n", __LINE__,               \
              error_string);                                                   \
      fflush(stderr);                                                          \
      MPI_Abort(MPI_COMM_WORLD, err_code);                                     \
    }                                                                          \
  }
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__,   \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define COLL_BASE_COMPUTE_BLOCKCOUNT(COUNT, NUM_BLOCKS, SPLIT_INDEX,           \
                                     EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT)      \
  EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;                   \
  SPLIT_INDEX = COUNT % NUM_BLOCKS;                                            \
  if (0 != SPLIT_INDEX) {                                                      \
    EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                                 \
  }

int allreduce_ring_comprs_hom_sum(const float *d_sbuf, float *d_rbuf,
                                  size_t count, MPI_Comm comm, float eb) {
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  int bsize, gsize;
  unsigned char *d_cmpReduceBytes;
  float *d_rtmpbuf;

  int *d_quant_predData;
  int early_segcount, late_segcount, split_rank, max_segcount;
  unsigned char *d_inbuf[2];
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Status status;
  int count_;

  cudaStream_t quant_prediction_stream;
  cudaStreamCreate(&quant_prediction_stream);

  if (1 == size) {
    return MPI_SUCCESS;
  }

  size_t pad_nbEle = (count + 32768 - 1) / 32768 * 32768;
  COLL_BASE_COMPUTE_BLOCKCOUNT(pad_nbEle, size, split_rank, early_segcount,
                               late_segcount);
  early_segcount = (early_segcount % 4 == 0)
                       ? early_segcount
                       : early_segcount + (4 - early_segcount % 4);
  late_segcount = (late_segcount % 4 == 0)
                      ? late_segcount
                      : late_segcount + (4 - late_segcount % 4);

  max_segcount = early_segcount;
  max_real_segsize = max_segcount * sizeof(float);

  size_t padded_count =
      early_segcount * split_rank + late_segcount * (size - split_rank);
  CUDA_CHECK(cudaMalloc((void **)&d_rtmpbuf, padded_count * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_quant_predData, max_segcount * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, max_segcount * sizeof(float)));

  cudaMalloc((void **)&d_inbuf[0], max_real_segsize);
  cudaMalloc((void **)&d_inbuf[1], max_real_segsize);

  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  inbi = 0;
  block_offset = ((rank < split_rank)
                      ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
                      : (ptrdiff_t)split_rank * early_segcount +
                            ((ptrdiff_t)rank - split_rank) * late_segcount);
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);

  GSZ_compress_deviceptr_outlier(d_rtmpbuf + block_offset, d_cmpReduceBytes,
                                 block_count, &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from,
                           0, comm, &reqs[inbi]));
  MPI_call_check(MPI_Send(d_cmpReduceBytes, cmpSize + (cmpSize * 0.1), MPI_BYTE,
                          send_to, 0, comm));
  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;
    block_offset =
        ((prevblock < split_rank)
             ? ((ptrdiff_t)prevblock * (ptrdiff_t)early_segcount)
             : (ptrdiff_t)split_rank * early_segcount +
                   ((ptrdiff_t)prevblock - split_rank) * late_segcount);
    block_count = ((prevblock < split_rank) ? early_segcount : late_segcount);
    bsize = dec_tblock_size;
    gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);

    dim3 grid(gsize);
    dim3 block(bsize);
    kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
        d_rtmpbuf + block_offset, d_quant_predData, eb, block_count);

    MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE,
                             recv_from, 0, comm, &reqs[inbi]));

    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));

    cudaStreamSynchronize(quant_prediction_stream);
    homomorphic_sum(d_inbuf[inbi ^ 0x1], d_quant_predData, d_cmpReduceBytes,
                    block_count, eb, &cmpSize);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(MPI_Send(d_cmpReduceBytes, cmpSize + (cmpSize * 0.1),
                            MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset =
      ((recv_from < split_rank)
           ? ((ptrdiff_t)recv_from * (ptrdiff_t)early_segcount)
           : (ptrdiff_t)split_rank * early_segcount +
                 ((ptrdiff_t)recv_from - split_rank) * late_segcount);
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);

  kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
      d_rtmpbuf + block_offset, d_quant_predData, eb, block_count);
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  cudaStreamSynchronize(quant_prediction_stream);

  homomorphic_sum(d_inbuf[inbi], d_quant_predData, d_inbuf[inbi ^ 0x1],
                  block_count, eb, &cmpSize);
  CUDA_CHECK(cudaGetLastError());
  GSZ_decompress_deviceptr_outlier(
      d_rtmpbuf + block_offset, d_inbuf[inbi ^ 0x1], block_count, cmpSize, eb);
  cmpSize = cmpSize + (cmpSize * 0.1);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    inbi = inbi ^ 0x1;
    const int recv_data_from = (rank + size - k) % size;
    const ptrdiff_t recv_block_offset =
        ((recv_data_from < split_rank)
             ? ((ptrdiff_t)recv_data_from * early_segcount)
             : (ptrdiff_t)split_rank * early_segcount +
                   ((ptrdiff_t)recv_data_from - split_rank) * late_segcount);
    block_count =
        ((recv_data_from < split_rank) ? early_segcount : late_segcount);

    MPI_call_check(MPI_Sendrecv(d_inbuf[inbi], cmpSize, MPI_BYTE, send_to, 0,
                                d_inbuf[inbi ^ 0x1], max_real_segsize, MPI_BYTE,
                                recv_from, 0, comm, &status));

    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = count_;
    GSZ_decompress_deviceptr_outlier(d_rtmpbuf + recv_block_offset,
                                     d_inbuf[inbi ^ 0x1], (size_t)block_count,
                                     cmpSize, eb);
    CUDA_CHECK(cudaGetLastError());
  }
  cudaMemcpy(d_rbuf, d_rtmpbuf, count * sizeof(float),
             cudaMemcpyDeviceToDevice);

  cudaFree(d_rtmpbuf);
  cudaFree(d_quant_predData);
  cudaFree(d_cmpReduceBytes);
  cudaFree(d_inbuf[0]);
  cudaFree(d_inbuf[1]);
  return 0;
}

int allreduce_ring_comprs_hom_sum_F(const float *d_sbuf, float *d_rbuf,
                                    size_t count, MPI_Comm comm, float eb) {
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  int bsize, gsize;
  unsigned char *d_cmpReduceBytes;
  float *d_rtmpbuf;

  int *d_quant_predData;
  int early_segcount, late_segcount, split_rank, max_segcount;
  unsigned char *d_inbuf[2];
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Status status;
  int count_;

  if (1 == size) {
    return MPI_SUCCESS;
  }

  size_t pad_nbEle = (count + 32768 - 1) / 32768 * 32768;
  COLL_BASE_COMPUTE_BLOCKCOUNT(pad_nbEle, size, split_rank, early_segcount,
                               late_segcount);
  early_segcount = (early_segcount % 4 == 0)
                       ? early_segcount
                       : early_segcount + (4 - early_segcount % 4);
  late_segcount = (late_segcount % 4 == 0)
                      ? late_segcount
                      : late_segcount + (4 - late_segcount % 4);

  max_segcount = early_segcount;
  max_real_segsize = max_segcount * sizeof(float);

  size_t padded_count =
      early_segcount * split_rank + late_segcount * (size - split_rank);
  CUDA_CHECK(cudaMalloc((void **)&d_rtmpbuf, padded_count * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, max_segcount * sizeof(float)));

  cudaMalloc((void **)&d_inbuf[0], max_real_segsize);
  cudaMalloc((void **)&d_inbuf[1], max_real_segsize);

  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  inbi = 0;
  block_offset = ((rank < split_rank)
                      ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
                      : (ptrdiff_t)split_rank * early_segcount +
                            ((ptrdiff_t)rank - split_rank) * late_segcount);
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);

  GSZ_compress_deviceptr_outlier(d_rtmpbuf + block_offset, d_cmpReduceBytes,
                                 block_count, &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from,
                           0, comm, &reqs[inbi]));
  MPI_call_check(MPI_Send(d_cmpReduceBytes, cmpSize + (cmpSize * 0.1), MPI_BYTE,
                          send_to, 0, comm));
  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;
    block_offset =
        ((prevblock < split_rank)
             ? ((ptrdiff_t)prevblock * (ptrdiff_t)early_segcount)
             : (ptrdiff_t)split_rank * early_segcount +
                   ((ptrdiff_t)prevblock - split_rank) * late_segcount);
    block_count = ((prevblock < split_rank) ? early_segcount : late_segcount);
    MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE,
                             recv_from, 0, comm, &reqs[inbi]));

    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));

    homomorphic_sum_F(d_inbuf[inbi ^ 0x1], d_rtmpbuf + block_offset,
                      d_cmpReduceBytes, block_count, eb, &cmpSize);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(MPI_Send(d_cmpReduceBytes, cmpSize + (cmpSize * 0.1),
                            MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset =
      ((recv_from < split_rank)
           ? ((ptrdiff_t)recv_from * (ptrdiff_t)early_segcount)
           : (ptrdiff_t)split_rank * early_segcount +
                 ((ptrdiff_t)recv_from - split_rank) * late_segcount);
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  homomorphic_sum_F(d_inbuf[inbi], d_rtmpbuf + block_offset,
                    d_inbuf[inbi ^ 0x1], block_count, eb, &cmpSize);
  CUDA_CHECK(cudaGetLastError());
  GSZ_decompress_deviceptr_outlier(
      d_rtmpbuf + block_offset, d_inbuf[inbi ^ 0x1], block_count, cmpSize, eb);
  cmpSize = cmpSize + (cmpSize * 0.1);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    inbi = inbi ^ 0x1;
    const int recv_data_from = (rank + size - k) % size;
    const ptrdiff_t recv_block_offset =
        ((recv_data_from < split_rank)
             ? ((ptrdiff_t)recv_data_from * early_segcount)
             : (ptrdiff_t)split_rank * early_segcount +
                   ((ptrdiff_t)recv_data_from - split_rank) * late_segcount);
    block_count =
        ((recv_data_from < split_rank) ? early_segcount : late_segcount);

    MPI_call_check(MPI_Sendrecv(d_inbuf[inbi], cmpSize, MPI_BYTE, send_to, 0,
                                d_inbuf[inbi ^ 0x1], max_real_segsize, MPI_BYTE,
                                recv_from, 0, comm, &status));

    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = count_;
    GSZ_decompress_deviceptr_outlier(d_rtmpbuf + recv_block_offset,
                                     d_inbuf[inbi ^ 0x1], (size_t)block_count,
                                     cmpSize, eb);
    CUDA_CHECK(cudaGetLastError());
  }
  cudaMemcpy(d_rbuf, d_rtmpbuf, count * sizeof(float),
             cudaMemcpyDeviceToDevice);

  cudaFree(d_rtmpbuf);
  cudaFree(d_cmpReduceBytes);
  cudaFree(d_inbuf[0]);
  cudaFree(d_inbuf[1]);
  return 0;
}