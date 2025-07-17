#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
  EARLY_BLOCK_COUNT = COUNT / NUM_BLOCKS;                                      \
  LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;                                       \
  SPLIT_INDEX = COUNT % NUM_BLOCKS;                                            \
  if (0 != SPLIT_INDEX) {                                                      \
    EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                                 \
  }

int cpuCopy_allreduce_ring_comprs_hom_sum(const float *d_sbuf, float *d_rbuf,
                                          size_t count, MPI_Comm comm,
                                          float eb) {
  int rank, size, k, recv_from, send_to, block_count, inbi, count_;
  int bsize, gsize;

  unsigned char *cmpReduceBytes;
  unsigned char *d_cmpReduceBytes;
  unsigned char *inbuf[2];
  unsigned char *d_tmpbuf;

  cudaStream_t quant_prediction_stream;
  CUDA_CHECK(cudaStreamCreate(&quant_prediction_stream));

  int *d_quant_predData;
  float *d_rtmpbuf;
  ptrdiff_t block_offset_elements;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Status status;

  if (1 == size) {

    CUDA_CHECK(cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return MPI_SUCCESS;
  }

  block_count = ceil(count / size);
  block_count = (block_count + 32768 - 1) / 32768 * 32768;

  // Host memory allocation
  cmpReduceBytes = (unsigned char *)malloc(block_count * sizeof(float));
  inbuf[0] = (unsigned char *)malloc(block_count * sizeof(float));
  inbuf[1] = (unsigned char *)malloc(block_count * sizeof(float));
  // device memory allocation
  CUDA_CHECK(
      cudaMalloc((void **)&d_rtmpbuf, block_count * size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_quant_predData, block_count * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, block_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_tmpbuf, block_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  inbi = 0;

  block_offset_elements = block_count * rank;

  float *d_rbuf_ = d_rtmpbuf + block_offset_elements;

  GSZ_compress_deviceptr_outlier(d_rbuf_, d_cmpReduceBytes, block_count,
                                 &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(cmpReduceBytes, d_cmpReduceBytes, cmpSize + 32768,
                        cudaMemcpyDeviceToHost));

  MPI_call_check(MPI_Irecv(inbuf[inbi], block_count * sizeof(float), MPI_BYTE,
                           recv_from, 0, comm, &reqs[inbi]));
  MPI_call_check(
      MPI_Send(cmpReduceBytes, cmpSize + 32768, MPI_BYTE, send_to, 0, comm));

  for (k = 2; k < size; k++) {

    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;
    block_offset_elements = block_count * prevblock;
    bsize = dec_tblock_size;
    gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    dim3 grid(gsize);
    dim3 block(bsize);
    d_rbuf_ = d_rtmpbuf + block_offset_elements;
    kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
        d_rbuf_, d_quant_predData, eb, block_count);
    MPI_call_check(MPI_Irecv(inbuf[inbi], block_count * sizeof(float), MPI_BYTE,
                             recv_from, 0, comm, &reqs[inbi]));
    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    cudaStreamSynchronize(quant_prediction_stream);
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = (size_t)count_;
    CUDA_CHECK(cudaMemcpy(d_tmpbuf, inbuf[inbi ^ 0x1], cmpSize,
                          cudaMemcpyHostToDevice));
    homomorphic_sum(d_tmpbuf, d_quant_predData, d_cmpReduceBytes, block_count,
                    eb, &cmpSize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(cmpReduceBytes, d_cmpReduceBytes, cmpSize + 32768,
                          cudaMemcpyDeviceToHost));

    MPI_call_check(
        MPI_Send(cmpReduceBytes, cmpSize + 32768, MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset_elements = block_count * recv_from;
  bsize = dec_tblock_size;
  gsize = (block_count + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);
  d_rbuf_ = d_rtmpbuf + block_offset_elements;
  kernel_quant_prediction<<<grid, block, 0, quant_prediction_stream>>>(
      d_rbuf_, d_quant_predData, eb, block_count);
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_call_check(MPI_Get_count(&status, MPI_BYTE, &count_));
  cmpSize = (size_t)count_;
  cudaStreamSynchronize(quant_prediction_stream);
  CUDA_CHECK(
      cudaMemcpy(d_tmpbuf, inbuf[inbi], cmpSize, cudaMemcpyHostToDevice));
  homomorphic_sum(d_tmpbuf, d_quant_predData, d_cmpReduceBytes, block_count, eb,
                  &cmpSize);
  cmpSize += 32768;
  GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset_elements,
                                   d_cmpReduceBytes, block_count, cmpSize, eb);
  cudaMemcpy(inbuf[inbi ^ 0x1], d_cmpReduceBytes,
             cmpSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    inbi = inbi ^ 0x1;
    block_offset_elements = block_count * ((rank + size - k) % size);
    memset(inbuf[inbi ^ 0x1], 0, block_count * sizeof(float));
    MPI_call_check(MPI_Sendrecv(inbuf[inbi], cmpSize, MPI_BYTE, send_to, 0,
                                inbuf[inbi ^ 0x1], block_count * sizeof(float),
                                MPI_BYTE, recv_from, 0, comm, &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = (size_t)count_;
    cudaMemset(d_cmpReduceBytes, 0, block_count * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_cmpReduceBytes, inbuf[inbi ^ 0x1], cmpSize,
                          cudaMemcpyHostToDevice));
    GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset_elements,
                                     d_cmpReduceBytes, (size_t)block_count,
                                     cmpSize, eb);
  }

  CUDA_CHECK(cudaMemcpy(d_rbuf, d_rtmpbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  free(cmpReduceBytes);
  free(inbuf[0]);
  free(inbuf[1]);
  CUDA_CHECK(cudaFree(d_rtmpbuf));
  CUDA_CHECK(cudaFree(d_quant_predData));
  CUDA_CHECK(cudaFree(d_cmpReduceBytes));
  CUDA_CHECK(cudaFree(d_tmpbuf));

  return 0;
}

int cpuCopy_allreduce_ring_comprs_hom_sum_F(const float *d_sbuf, float *d_rbuf,
                                            size_t count, MPI_Comm comm,
                                            float eb) {
  int rank, size, k, recv_from, send_to, block_count, inbi, count_;

  unsigned char *cmpReduceBytes;
  unsigned char *d_cmpReduceBytes;
  unsigned char *inbuf[2];
  unsigned char *d_tmpbuf;

  float *d_rtmpbuf;
  ptrdiff_t block_offset_elements;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Status status;

  if (1 == size) {

    CUDA_CHECK(cudaMemcpy(d_rbuf, d_sbuf, count * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return MPI_SUCCESS;
  }
  block_count = ceil(count / size);
  block_count = (block_count + 32768 - 1) / 32768 * 32768;

  // Host memory allocation
  cmpReduceBytes = (unsigned char *)malloc(block_count * sizeof(float));
  inbuf[0] = (unsigned char *)malloc(block_count * sizeof(float));
  inbuf[1] = (unsigned char *)malloc(block_count * sizeof(float));

  // device memory allocation
  CUDA_CHECK(
      cudaMalloc((void **)&d_rtmpbuf, block_count * size * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_cmpReduceBytes, block_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_tmpbuf, block_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_rtmpbuf, d_sbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  size_t cmpSize;
  inbi = 0;

  block_offset_elements = (ptrdiff_t)block_count * rank;
  float *d_rbuf_ = d_rtmpbuf + block_offset_elements;

  GSZ_compress_deviceptr_outlier(d_rbuf_, d_cmpReduceBytes, block_count,
                                 &cmpSize, eb);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(cmpReduceBytes, d_cmpReduceBytes, cmpSize + 32768,
                        cudaMemcpyDeviceToHost));

  MPI_call_check(MPI_Irecv(inbuf[inbi], block_count * sizeof(float), MPI_BYTE,
                           recv_from, 0, comm, &reqs[inbi]));
  MPI_call_check(
      MPI_Send(cmpReduceBytes, cmpSize + 32768, MPI_BYTE, send_to, 0, comm));

  for (k = 2; k < size; k++) {

    const int prevblock = (rank + size - k + 1) % size;
    inbi = inbi ^ 0x1;

    block_offset_elements = (ptrdiff_t)block_count * prevblock;

    d_rbuf_ = d_rtmpbuf + block_offset_elements;
    MPI_call_check(MPI_Irecv(inbuf[inbi], block_count * sizeof(float), MPI_BYTE,
                             recv_from, 0, comm, &reqs[inbi]));
    MPI_call_check(MPI_Wait(&reqs[inbi ^ 0x1], &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = (size_t)count_;

    CUDA_CHECK(cudaMemcpy(d_tmpbuf, inbuf[inbi ^ 0x1], cmpSize,
                          cudaMemcpyHostToDevice));

    homomorphic_sum_F(d_tmpbuf, d_rbuf_, d_cmpReduceBytes, block_count, eb,
                      &cmpSize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(cmpReduceBytes, d_cmpReduceBytes, cmpSize + 32768,
                          cudaMemcpyDeviceToHost));

    MPI_call_check(
        MPI_Send(cmpReduceBytes, cmpSize + 32768, MPI_BYTE, send_to, 0, comm));
  }
  recv_from = (rank + 1) % size;
  block_offset_elements = (ptrdiff_t)block_count * recv_from;
  d_rbuf_ = d_rtmpbuf + block_offset_elements;
  MPI_call_check(MPI_Wait(&reqs[inbi], &status));
  MPI_call_check(MPI_Get_count(&status, MPI_BYTE, &count_));
  cmpSize = (size_t)count_;
  CUDA_CHECK(cudaMemcpy(d_tmpbuf, inbuf[inbi], cmpSize + 32768,
                        cudaMemcpyHostToDevice));

  homomorphic_sum_F(d_tmpbuf, d_rbuf_, d_cmpReduceBytes, block_count, eb,
                    &cmpSize);
  cmpSize += 32768;
  GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset_elements,
                                   d_cmpReduceBytes, block_count, cmpSize, eb);
  cudaMemcpy(inbuf[inbi ^ 0x1], d_cmpReduceBytes, cmpSize,
             cudaMemcpyDeviceToHost);
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    inbi = inbi ^ 0x1;
    const int recv_data_from = (rank + size - k) % size;
    block_offset_elements = (ptrdiff_t)block_count * recv_data_from;
    MPI_call_check(MPI_Sendrecv(inbuf[inbi], cmpSize, MPI_BYTE, send_to, 0,
                                inbuf[inbi ^ 0x1], block_count * sizeof(float),
                                MPI_BYTE, recv_from, 0, comm, &status));
    MPI_Get_count(&status, MPI_BYTE, &count_);
    cmpSize = (size_t)count_;
    CUDA_CHECK(cudaMemcpy(d_cmpReduceBytes, inbuf[inbi ^ 0x1], cmpSize,
                          cudaMemcpyHostToDevice));
    GSZ_decompress_deviceptr_outlier(d_rtmpbuf + block_offset_elements,
                                     d_cmpReduceBytes, (size_t)block_count,
                                     cmpSize, eb);
  }

  CUDA_CHECK(cudaMemcpy(d_rbuf, d_rtmpbuf, count * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  free(cmpReduceBytes);
  free(inbuf[0]);
  free(inbuf[1]);
  CUDA_CHECK(cudaFree(d_rtmpbuf));
  CUDA_CHECK(cudaFree(d_cmpReduceBytes));
  CUDA_CHECK(cudaFree(d_tmpbuf));

  return 0;
}