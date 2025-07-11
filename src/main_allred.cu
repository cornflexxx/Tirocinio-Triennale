#include "../include/AWcudaAllreduce.cuh"
#include <cstddef>
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
  if (argc < 2) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <iterations>\n", argv[0]);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  int iterations = atoi(argv[1]);
  if (iterations <= 0) {
    if (rank == 0) {
      fprintf(stderr, "Iterations must be a positive integer.\n");
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(0));
  float *h_sbuf;
  h_sbuf = read_data("smooth.in", &count);
  float *h_rbuf = (float *)malloc(count * sizeof(float));
  float *d_sbuf, *d_rbuf;
  cudaMalloc((void **)&d_sbuf, count * sizeof(float));
  cudaMemcpy(d_sbuf, h_sbuf, count * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_rbuf, count * sizeof(float));
  double MPI_timer = 0.0;
  float eb = 0.0001;
  for (int i = 0; i < iterations; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_timer -= MPI_Wtime();
    allreduce_ring_comprs_hom_sum_F(d_sbuf, d_rbuf, count, MPI_COMM_WORLD, eb);
    // MPI_Allreduce(d_sbuf,d_rbuf, count,MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_timer += MPI_Wtime();
  }
  double latency = MPI_timer / iterations;
  double min_time = 0.0;
  double max_time = 0.0;
  double avg_time = 0.0;
  MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  avg_time = avg_time / size;

  cudaMemcpy(h_rbuf, d_rbuf, count * sizeof(float), cudaMemcpyDeviceToHost);
  if (rank == 0) {
    printf("Min time: %f seconds\n", min_time);
    printf("Max time: %f seconds\n", max_time);
    printf("Avg time: %f seconds\n", avg_time);
    printf("Iterations: %d\n", iterations);
    printf("Count: %zu\n", count);
    write_dataf("smooth.out", h_rbuf, count);
  }

  cudaFree(d_sbuf);
  cudaFree(d_rbuf);
  MPI_Finalize();
}
