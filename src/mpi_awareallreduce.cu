#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>

ptrdiff_t datatype_span(MPI_Datatype dtype, size_t count, ptrdiff_t *gap) {
  int ret;
  ptrdiff_t lb, extent, true_lb, true_extent;

  ret = MPI_Type_get_extent(dtype, &lb, &extent);
  if (MPI_SUCCESS != ret) {
    return -1;
  }
  ret = MPI_Type_get_true_extent(dtype, &true_lb, &true_extent);
  if (MPI_SUCCESS != ret) {
    return -1;
  }

  *gap = true_lb - lb;
  return (ptrdiff_t)(count * extent + *gap);
}

int sizeof_datatype(MPI_Datatype dtype) {
  int size;
  MPI_Type_size(dtype, &size);
  return size;
}
#define GPUS_PER_NODE 4

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
  int rank, size, k, recv_from, send_to, block_count, inbi;
  int bsize, gsize;
  unsigned char *d_cmpReduceBytes;
  float *d_rtmpbuf;

  int *d_quant_predData;
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

  block_count = ceil(count / size);
  block_count = (block_count + 32768 - 1) / 32768 * 32768;

  CUDA_CHECK(
      cudaMalloc((void **)&d_rtmpbuf, block_count * size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_quant_predData, block_count * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, block_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[0], block_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[1], block_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  size_t cmpSize2;
  inbi = 0;
  block_offset = block_count * rank;
  GSZ_compress_deviceptr_outlier(d_rtmpbuf + block_offset, d_cmpReduceBytes,
                                 block_count, &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());

  MPI_call_check(MPI_Irecv(d_inbuf[inbi], block_count * sizeof(float), MPI_BYTE,
                           recv_from, 0, comm, &reqs[inbi]));
  MPI_call_check(
      MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));

  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;
    block_offset = block_count * prevblock;
    bsize = dec_tblock_size;
    gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);

    dim3 grid(gsize);
    dim3 block(bsize);
    kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
        d_rtmpbuf + block_offset, d_quant_predData, eb, block_count);

    MPI_call_check(MPI_Irecv(d_inbuf[inbi], block_count * sizeof(float),
                             MPI_BYTE, recv_from, 0, comm, &reqs[inbi]));
    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);

    cmpSize2 = count_;
    cudaStreamSynchronize(quant_prediction_stream);

    homomorphic_sum(d_inbuf[inbi ^ 0x1], d_quant_predData, d_cmpReduceBytes,
                    block_count, eb, &cmpSize, cmpSize2);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(
        MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset = block_count * recv_from;
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);
  kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
      d_rtmpbuf + block_offset, d_quant_predData, eb, block_count);
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_Get_count(&status, MPI_BYTE, &count_);
  cmpSize2 = count_;
  cudaStreamSynchronize(quant_prediction_stream);

  homomorphic_sum(d_inbuf[inbi], d_quant_predData, d_inbuf[inbi ^ 0x1],
                  block_count, eb, &cmpSize, cmpSize2);
  CUDA_CHECK(cudaGetLastError());
  GSZ_decompress_deviceptr_outlier(
      d_rtmpbuf + block_offset, d_inbuf[inbi ^ 0x1], block_count, cmpSize, eb);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    inbi = inbi ^ 0x1;
    const int recv_data_from = (rank + size - k) % size;
    block_offset = recv_data_from * block_count;
    MPI_call_check(MPI_Sendrecv(
        d_inbuf[inbi], cmpSize, MPI_BYTE, send_to, 0, d_inbuf[inbi ^ 0x1],
        block_count * sizeof(float), MPI_BYTE, recv_from, 0, comm, &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = count_;
    GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset,
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
  int rank, size, k, recv_from, send_to, block_count, inbi;
  unsigned char *d_cmpReduceBytes;
  float *d_rtmpbuf;

  unsigned char *d_inbuf[2];
  ptrdiff_t block_offset;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Status status;
  int count_;

  if (1 == size) {
    return MPI_SUCCESS;
  }

  block_count = ceil(count / size);
  block_count = (block_count + 32768 - 1) / 32768 * 32768;

  CUDA_CHECK(
      cudaMalloc((void **)&d_rtmpbuf, block_count * size * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, block_count * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[0], block_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[1], block_count * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize, cmpSize2;
  inbi = 0;
  block_offset = block_count * rank;

  GSZ_compress_deviceptr_outlier(d_rtmpbuf + block_offset, d_cmpReduceBytes,
                                 block_count, &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  MPI_call_check(MPI_Irecv(d_inbuf[inbi], block_count * sizeof(float), MPI_BYTE,
                           recv_from, 0, comm, &reqs[inbi]));
  MPI_call_check(
      MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;
    block_offset = block_count * prevblock;
    MPI_call_check(MPI_Irecv(d_inbuf[inbi], block_count * sizeof(float),
                             MPI_BYTE, recv_from, 0, comm, &reqs[inbi]));

    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize2 = count_;
    homomorphic_sum_F(d_inbuf[inbi ^ 0x1], d_rtmpbuf + block_offset,
                      d_cmpReduceBytes, block_count, eb, &cmpSize, cmpSize2);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(
        MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset = block_count * recv_from;
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_Get_count(&status, MPI_BYTE, &count_);
  cmpSize2 = count_;
  homomorphic_sum_F(d_inbuf[inbi], d_rtmpbuf + block_offset,
                    d_inbuf[inbi ^ 0x1], block_count, eb, &cmpSize, cmpSize2);
  CUDA_CHECK(cudaGetLastError());
  GSZ_decompress_deviceptr_outlier(
      d_rtmpbuf + block_offset, d_inbuf[inbi ^ 0x1], block_count, cmpSize, eb);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    inbi = inbi ^ 0x1;
    const int recv_data_from = (rank + size - k) % size;
    block_offset = block_count * recv_data_from;
    MPI_call_check(MPI_Sendrecv(
        d_inbuf[inbi], cmpSize, MPI_BYTE, send_to, 0, d_inbuf[inbi ^ 0x1],
        block_count * sizeof(float), MPI_BYTE, recv_from, 0, comm, &status));

    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = count_;
    GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset,
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

int allreduce_ring_comprs_hom_sum_seg(const float *d_sbuf, float *d_rbuf,
                                      size_t count, MPI_Comm comm, float eb) {
  int rank, size, k, recv_from, send_to, block_count, inbi;
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
  padded_count = (padded_count + 32768 - 1) / 32768 * 32768;
  CUDA_CHECK(cudaMalloc((void **)&d_rtmpbuf, padded_count * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_quant_predData, max_segcount * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, max_segcount * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[0], max_real_segsize));
  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[1], max_real_segsize));

  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  size_t cmpSize2;
  inbi = 0;
  block_offset = ((rank < split_rank)
                      ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
                      : (ptrdiff_t)split_rank * early_segcount +
                            ((ptrdiff_t)rank - split_rank) * late_segcount);
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);
  if (rank == size - 1)
    block_count = count - block_offset;
  GSZ_compress_deviceptr_outlier(d_rtmpbuf + block_offset, d_cmpReduceBytes,
                                 block_count, &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from,
                           0, comm, &reqs[inbi]));
  MPI_call_check(
      MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
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
    if (prevblock == size - 1)
      block_count = count - block_offset;
    dim3 grid(gsize);
    dim3 block(bsize);
    kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
        d_rtmpbuf + block_offset, d_quant_predData, eb, block_count);

    MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE,
                             recv_from, 0, comm, &reqs[inbi]));

    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);

    cmpSize2 = count_;
    cudaStreamSynchronize(quant_prediction_stream);
    homomorphic_sum(d_inbuf[inbi ^ 0x1], d_quant_predData, d_cmpReduceBytes,
                    block_count, eb, &cmpSize, cmpSize2);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(
        MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset =
      ((recv_from < split_rank)
           ? ((ptrdiff_t)recv_from * (ptrdiff_t)early_segcount)
           : (ptrdiff_t)split_rank * early_segcount +
                 ((ptrdiff_t)recv_from - split_rank) * late_segcount);
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  if (recv_from == size - 1)
    block_count = count - block_offset;
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);

  kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
      d_rtmpbuf + block_offset, d_quant_predData, eb, block_count);
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_Get_count(&status, MPI_BYTE, &count_);
  cmpSize2 = count_;
  cudaStreamSynchronize(quant_prediction_stream);

  homomorphic_sum(d_inbuf[inbi], d_quant_predData, d_inbuf[inbi ^ 0x1],
                  block_count, eb, &cmpSize, cmpSize2);
  CUDA_CHECK(cudaGetLastError());
  GSZ_decompress_deviceptr_outlier(
      d_rtmpbuf + block_offset, d_inbuf[inbi ^ 0x1], block_count, cmpSize, eb);
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
    if (recv_data_from == size - 1)
      block_count = count - recv_block_offset;
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

int allreduce_ring_comprs_hom_sum_F_seg(const float *d_sbuf, float *d_rbuf,
                                        size_t count, MPI_Comm comm, float eb) {
  int rank, size, k, recv_from, send_to, block_count, inbi;
  unsigned char *d_cmpReduceBytes;
  float *d_rtmpbuf;
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
  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[0], max_real_segsize));
  CUDA_CHECK(cudaMalloc((void **)&d_inbuf[1], max_real_segsize));
  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize, cmpSize2;
  inbi = 0;
  block_offset = ((rank < split_rank)
                      ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
                      : (ptrdiff_t)split_rank * early_segcount +
                            ((ptrdiff_t)rank - split_rank) * late_segcount);
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);
  if (rank == size - 1)
    block_count = count - block_offset;
  GSZ_compress_deviceptr_outlier(d_rtmpbuf + block_offset, d_cmpReduceBytes,
                                 block_count, &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE, recv_from,
                           0, comm, &reqs[inbi]));
  MPI_call_check(
      MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;
    block_offset =
        ((prevblock < split_rank)
             ? ((ptrdiff_t)prevblock * (ptrdiff_t)early_segcount)
             : (ptrdiff_t)split_rank * early_segcount +
                   ((ptrdiff_t)prevblock - split_rank) * late_segcount);
    block_count = ((prevblock < split_rank) ? early_segcount : late_segcount);
    if (prevblock == size - 1)
      block_count = count - block_offset;
    MPI_call_check(MPI_Irecv(d_inbuf[inbi], max_real_segsize, MPI_BYTE,
                             recv_from, 0, comm, &reqs[inbi]));

    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize2 = count_;
    homomorphic_sum_F(d_inbuf[inbi ^ 0x1], d_rtmpbuf + block_offset,
                      d_cmpReduceBytes, block_count, eb, &cmpSize, cmpSize2);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(
        MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset =
      ((recv_from < split_rank)
           ? ((ptrdiff_t)recv_from * (ptrdiff_t)early_segcount)
           : (ptrdiff_t)split_rank * early_segcount +
                 ((ptrdiff_t)recv_from - split_rank) * late_segcount);
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  if (recv_from == size - 1)
    block_count = count - block_offset;
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_Get_count(&status, MPI_BYTE, &count_);
  cmpSize2 = count_;
  homomorphic_sum_F(d_inbuf[inbi], d_rtmpbuf + block_offset,
                    d_inbuf[inbi ^ 0x1], block_count, eb, &cmpSize, cmpSize2);
  CUDA_CHECK(cudaGetLastError());
  GSZ_decompress_deviceptr_outlier(
      d_rtmpbuf + block_offset, d_inbuf[inbi ^ 0x1], block_count, cmpSize, eb);
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
    if (recv_data_from == size - 1)
      block_count = count - recv_block_offset;
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

__global__ void sum4arrays(float *__restrict__ a, const float *__restrict__ b,
                           const float *__restrict__ c,
                           const float *__restrict__ d, size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < count; i += stride) {
    a[i] += b[i] + c[i] + d[i];
  }
}

int intra_node_reduce_scatter(float *d_sbuf, float *d_rbuf, size_t count,
                              MPI_Comm local_comm) {
  int rank, size;
  MPI_Comm_rank(local_comm, &rank);
  MPI_Comm_size(local_comm, &size);

  if (size == 1) {
    cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float), cudaMemcpyDeviceToDevice);
    return MPI_SUCCESS;
  }

  MPI_Request *recv_reqs = (MPI_Request *)malloc(size * sizeof(MPI_Request));
  size_t send_count;
  size_t per_rank_count = count / size;

  for (int i = 0; i < size; i++) {
    send_count = (i == size - 1) ? count - (per_rank_count)*i : per_rank_count;

    if (i != rank) {
      MPI_Isend(d_sbuf + i * per_rank_count, send_count, MPI_FLOAT, i, 0,
                local_comm, MPI_REQUEST_NULL);
      MPI_Irecv(d_rbuf + i * per_rank_count, send_count, MPI_FLOAT, i, 0,
                local_comm, &recv_reqs[i]);
    }
  }
  MPI_Waitall(size, recv_reqs, MPI_STATUSES_IGNORE);

  int threads = 1024;
  int blocks = (count + threads - 1) / threads;
  send_count =
      (rank == size - 1) ? count - (per_rank_count)*rank : per_rank_count;

  sum4arrays<<<blocks, threads>>>(
      d_rbuf + rank * per_rank_count, d_rbuf + per_rank_count,
      d_rbuf + per_rank_count * 2, d_rbuf + per_rank_count * 3, send_count);

  free(recv_reqs);
  return MPI_SUCCESS;
}

int intra_node_allgather(float *d_rbuf, float *d_sbuf, size_t count,
                         MPI_Comm local_comm) {
  int rank, size;
  MPI_Comm_rank(local_comm, &rank);
  MPI_Comm_size(local_comm, &size);
  if (size == 1) {
    cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float), cudaMemcpyDeviceToDevice);
    return MPI_SUCCESS;
  }
  size_t per_rank_count = count / size;
  size_t send_count;
  MPI_Request *recv_reqs = (MPI_Request *)malloc(size * sizeof(MPI_Request));
  for (int i = 0; i < size; i++) {
    send_count = (i == size - 1) ? count - (per_rank_count)*i : per_rank_count;
    if (i != rank) {
      MPI_Isend(d_rbuf + rank * per_rank_count, send_count, MPI_FLOAT, i, 0,
                local_comm, MPI_REQUEST_NULL);
      MPI_Irecv(d_rbuf + i * per_rank_count, send_count, MPI_FLOAT, i, 0,
                local_comm, &recv_reqs[i]);
    }
  }
  MPI_Waitall(size, recv_reqs, MPI_STATUSES_IGNORE);
  free(recv_reqs);
}

int mixed_compressed_allreduce(float *d_sbuf, float *d_rbuf, size_t count,
                               MPI_Comm comm, float eb) {

  int rank, size, local_rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (size == 1) {
    cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float), cudaMemcpyDeviceToDevice);
    return MPI_SUCCESS;
  }
  // intra-node comm
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);

  // inter-node comms
  MPI_Comm inter_comm;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_split(comm, local_rank, rank, &inter_comm);

  intra_node_reduce_scatter(d_sbuf, d_rbuf, count, local_comm);

  size_t per_rank_count = (local_rank == size - 1)
                              ? count - (count / size) * local_rank
                              : count / size;
  allreduce_ring_comprs_hom_sum_F(d_rbuf + local_rank * (count / size),
                                  d_rbuf + local_rank * (count / size),
                                  per_rank_count, inter_comm, eb);
  intra_node_allgather(d_rbuf, d_rbuf, count, local_comm);
  MPI_Comm_free(&inter_comm);
  MPI_Comm_free(&local_comm);
  return MPI_SUCCESS;
}

int allreduce_ring_gpu(const void *d_sbuf, void *d_rbuf, size_t count,
                       MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  void *ret_;
  int early_segcount, late_segcount, split_rank, max_segcount;
  char *d_tmpsend = NULL, *d_tmprecv = NULL, *d_inbuf[2] = {NULL, NULL};
  ptrdiff_t true_lb, true_extent, lb, extent;
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  ret = MPI_Comm_rank(comm, &rank); // get rank
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }
  ret = MPI_Comm_size(comm, &size); // get size of comm
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  // if(rank == 0) {
  //   printf("4: RING\n");
  //   fflush(stdout);
  // }

  /* Special case for size == 1 */
  if (1 == size) { // if only one rank, no need to do anything
    if (MPI_IN_PLACE != sbuf) {
      ret_ = cudaMemcpy((char *)d_sbuf, (char *)d_rbuf,
                        count * sizeof_datatype(dtype));
      if (ret_ < 0) {
        line = __LINE__;
        goto error_hndl;
      }
    }
    return MPI_SUCCESS;
  }

  /* Allocate and initialize temporary buffers */
  ret = MPI_Type_get_extent(dtype, &lb, &extent); // get extent of datatype
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }
  ret = MPI_Type_get_true_extent(dtype, &true_lb,
                                 &true_extent); // get true extent
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  /* Determine the number of elements per block and corresponding
  block sizes.
  The blocks are divided into "early" and "late" ones:
  blocks 0 .. (split_rank - 1) are "early" and
  blocks (split_rank) .. (size - 1) are "late".
  Early blocks are at most 1 element larger than the late ones.
  */
  // compute the block with +1 element respect to the other
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank, early_segcount,
                               late_segcount);
  max_segcount = early_segcount;
  max_real_segsize = true_extent + (max_segcount - 1) * extent;

  cudaMalloc((void **)&d_inbuf[0], max_real_segsize);
  if (size > 2) {
    cudaMalloc((void **)&d_inbuf[1], max_real_segsize);
  }

  /* Handle MPI_IN_PLACE */
  if (MPI_IN_PLACE != d_sbuf) {
    ret_ = cudaMemcpy((char *)d_sbuf, (char *)d_rbuf,
                      count * sizeof_datatype(dtype));
    if (ret_ == NULL) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  /* Computation loop */

  /*
  For each of the remote nodes:
  - post irecv for block (r-1)
  - send block (r)
  - in loop for every step k = 2 .. n
  - post irecv for block (r + n - k) % n
  - wait on block (r + n - k + 1) % n to arrive
  - compute on block (r + n - k + 1) % n
  - send block (r + n - k + 1) % n
  - wait on block (r + 1)
  - compute on block (r + 1)
  - send block (r + 1) to rank (r + 1)
  Note that we must be careful when computing the beginning of buffers and
  for send operations and computation we must compute the exact block size.
  */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;

  inbi = 0;
  /* Initialize first receive from the neighbor on the left */
  ret = MPI_Irecv(d_inbuf[inbi], max_segcount, dtype, recv_from, 0, comm,
                  &reqs[inbi]);
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }
  /* Send first block (my block) to the neighbor on the right */
  block_offset =
      ((rank < split_rank)
           ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
           : ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);
  d_tmpsend = ((char *)d_rbuf) + block_offset * extent;
  ret = MPI_Send(d_tmpsend, block_count, dtype, send_to, 0, comm);
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;

    inbi = inbi ^ 0x1;

    /* Post irecv for the current block */
    ret = MPI_Irecv(d_inbuf[inbi], max_segcount, dtype, recv_from, 0, comm,
                    &reqs[inbi]);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }

    /* Wait on previous block to arrive */
    ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }

    /* Apply operation on previous block: result goes to rbuf
    rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
    */
    block_offset = ((prevblock < split_rank)
                        ? ((ptrdiff_t)prevblock * early_segcount)
                        : ((ptrdiff_t)prevblock * late_segcount + split_rank));
    block_count = ((prevblock < split_rank) ? early_segcount : late_segcount);
    d_tmprecv = ((char *)d_rbuf) + (ptrdiff_t)block_offset * extent;
    MPI_Reduce_local(d_inbuf[inbi ^ 0x1], d_tmprecv, block_count, dtype, op);

    /* send previous block to send_to */
    ret = MPI_Send(d_tmprecv, block_count, dtype, send_to, 0, comm);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  /* Wait on the last block to arrive */
  ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  /* Apply operation on the last block (from neighbor (rank + 1)
  rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
  recv_from = (rank + 1) % size;
  block_offset = ((recv_from < split_rank)
                      ? ((ptrdiff_t)recv_from * early_segcount)
                      : ((ptrdiff_t)recv_from * late_segcount + split_rank));
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  d_tmprecv = ((char *)d_rbuf) + (ptrdiff_t)block_offset * extent;
  MPI_Reduce_local(d_inbuf[inbi], d_tmprecv, block_count, dtype, op);

  /* Distribution loop - variation of ring allgather */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    const int recv_data_from = (rank + size - k) % size;
    const int send_data_from = (rank + 1 + size - k) % size;
    const int send_block_offset =
        ((send_data_from < split_rank)
             ? ((ptrdiff_t)send_data_from * early_segcount)
             : ((ptrdiff_t)send_data_from * late_segcount + split_rank));
    const int recv_block_offset =
        ((recv_data_from < split_rank)
             ? ((ptrdiff_t)recv_data_from * early_segcount)
             : ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
    block_count =
        ((send_data_from < split_rank) ? early_segcount : late_segcount);

    d_tmprecv = (char *)d_rbuf + (ptrdiff_t)recv_block_offset * extent;
    d_tmpsend = (char *)d_rbuf + (ptrdiff_t)send_block_offset * extent;

    ret = MPI_Sendrecv(d_tmpsend, block_count, dtype, send_to, 0, d_tmprecv,
                       max_segcount, dtype, recv_from, 0, comm,
                       MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  if (NULL != d_inbuf[0])
    free(d_inbuf[0]);
  if (NULL != d_inbuf[1])
    free(d_inbuf[1]);

  return MPI_SUCCESS;

error_hndl:
  fprintf(stderr, "\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line,
          rank, ret);
  MPI_Request_free(&reqs[0]);
  MPI_Request_free(&reqs[1]);
  (void)line; // silence compiler warning
  if (NULL != d_inbuf[0])
    free(d_inbuf[0]);
  if (NULL != d_inbuf[1])
    free(d_inbuf[1]);
  return ret;
}