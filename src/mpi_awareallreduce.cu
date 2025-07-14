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

  block_count = ceil(count / size);
  block_count = (block_count + 32768 - 1) / 32768 * 32768;

  CUDA_CHECK(
      cudaMalloc((void **)&d_rtmpbuf, block_count * size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_quant_predData, block_count * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, block_count * sizeof(float)));

  cudaMalloc((void **)&d_inbuf[0], block_count * sizeof(float));
  cudaMalloc((void **)&d_inbuf[1], block_count * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
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

    cudaStreamSynchronize(quant_prediction_stream);
    homomorphic_sum(d_inbuf[inbi ^ 0x1], d_quant_predData, d_cmpReduceBytes,
                    block_count, eb, &cmpSize);
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
  cudaStreamSynchronize(quant_prediction_stream);

  homomorphic_sum(d_inbuf[inbi], d_quant_predData, d_inbuf[inbi ^ 0x1],
                  block_count, eb, &cmpSize);
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

  cudaMalloc((void **)&d_inbuf[0], block_count * sizeof(float));
  cudaMalloc((void **)&d_inbuf[1], block_count * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
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

    homomorphic_sum_F(d_inbuf[inbi ^ 0x1], d_rtmpbuf + block_offset,
                      d_cmpReduceBytes, block_count, eb, &cmpSize);
    CUDA_CHECK(cudaGetLastError());

    MPI_call_check(
        MPI_Send(d_cmpReduceBytes, cmpSize, MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset = block_count * recv_from;
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  homomorphic_sum_F(d_inbuf[inbi], d_rtmpbuf + block_offset,
                    d_inbuf[inbi ^ 0x1], block_count, eb, &cmpSize);
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