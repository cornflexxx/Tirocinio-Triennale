#include "../include/nca_allreduce.cuh"
#include "../include/readFile.h"
#include <cstddef>
#include <cstdio>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__,   \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
float *readFloatData_systemEndian(const char *srcFilePath, size_t *nbEle) {
  size_t inSize;
  FILE *pFile = fopen(srcFilePath, "rb");
  if (pFile == NULL) {
    printf("Failed to open input file. 1\n");
    return NULL;
  }
  fseek(pFile, 0, SEEK_END);
  inSize = ftell(pFile);
  *nbEle = inSize / 4;
  fclose(pFile);

  if (inSize <= 0) {
    printf("Error: input file is wrong!\n");
  }

  float *daBuf = (float *)malloc(inSize);

  pFile = fopen(srcFilePath, "rb");
  if (pFile == NULL) {
    printf("Failed to open input file. 2\n");
    return NULL;
  }
  fread(daBuf, 4, *nbEle, pFile);
  fclose(pFile);
  return daBuf;
}
float *read_data(const char *filename, size_t *dim) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Err");
    return NULL;
  }

  size_t sz = 1000;
  *dim = 0;
  float *vec = (float *)malloc(sz * sizeof(float));
  if (!vec) {
    perror("mem allocation failed");
    fclose(file);
    return NULL;
  }

  char row[100];

  while (fgets(row, sizeof(row), file)) {
    if (*dim >= sz) {
      sz *= 2;
      float *temp = (float *)realloc(vec, sz * sizeof(float));
      if (!temp) {
        perror("mem allocation failed");
        free(vec);
        fclose(file);
        return NULL;
      }
      vec = temp;
    }
    vec[*dim] = strtof(row, NULL);
    (*dim)++;
  }

  fclose(file);
  return vec;
}

void write_dataf(const char *filename, float *data, size_t dim) {
  FILE *file = fopen(filename, "w");
  if (!file) {
    perror("Err");
    return;
  }

  for (size_t i = 0; i < dim; i++) {
    fprintf(file, "%f\n", data[i]);
  }

  fclose(file);
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  size_t count;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  CUDA_CHECK(cudaSetDevice(0));
  char *filename = argv[1];
  float *h_sbuf;
  // h_sbuf =
  // readFloatData_systemEndian("/leonardo_scratch/large/userexternal/fcarboni/SDRBENCH-EXASKY-NYX-512x512x512/velocity_x.f32",
  // &count);
  h_sbuf = read_binary_floats(filename, &count);
  float *h_rbuf = (float *)malloc(count * sizeof(float));
  float *d_sbuf, *d_rbuf;
  double t1, t2;

  cudaMalloc((void **)&d_sbuf, count * sizeof(float));
  cudaMalloc((void **)&d_rbuf, count * sizeof(float));
  cudaMemcpy(d_sbuf, h_sbuf, count * sizeof(float), cudaMemcpyHostToDevice);
  t1 = MPI_Wtime();
  float eb = 0.001;
  cpuCopy_allreduce_ring_comprs_hom_sum_F(d_sbuf, d_rbuf, count, MPI_COMM_WORLD,
                                          eb);
  t2 = MPI_Wtime();
  if (rank == 0) {
    printf("Time taken for allreduce: %f seconds\n", t2 - t1);
  }
  cudaMemcpy(h_rbuf, d_rbuf, count * sizeof(float), cudaMemcpyDeviceToHost);
  if (rank == 0) {
    write_dataf("smooth.out", h_rbuf, count);
  }
  MPI_Finalize();
}