#include "../include/mpi_awareallreduce.cuh"
#include "../include/nca_allreduce.cuh"
#include "../include/readFile.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__,   \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

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
int main(int argc, char *argv[]) {
  size_t nbEle = 0;
  int iterations = 10;
  char input_file[512] = "";
  int mode = 0; // 0: normal, 1: mixed, 2: opt
  int opt;
  float eb = 0.0001f; // Default error bound
  int option_index = 0;
  static struct option long_options[] = {
      {"iter", required_argument, 0, 'i'}, {"file", required_argument, 0, 'f'},
      {"mode", required_argument, 0, 'm'}, {"help", no_argument, 0, 'h'},
      {"eb", required_argument, 0, 'b'},   {0, 0, 0, 0}};
  while ((opt = getopt_long(argc, argv, "i:f:m:h", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'f':
      strncpy(input_file, optarg, sizeof(input_file) - 1);
      input_file[sizeof(input_file) - 1] = '\0';
      break;
    case 'm':
      if (strcmp(optarg, "normal") == 0) {
        mode = 0;
      } else if (strcmp(optarg, "mixed") == 0) {
        mode = 1;
      } else if (strcmp(optarg, "opt") == 0) {
        mode = 2;
      } else {
        fprintf(stderr, "Invalid mode: %s. Use 'normal', 'mixed', or 'opt'.\n",
                optarg);
        return EXIT_FAILURE;
      }
      break;
    case 'b':
      eb = atof(optarg);
      if (eb <= 0.0f) {
        fprintf(stderr, "Error bound must be a positive number.\n");
        return EXIT_FAILURE;
      }
      break;
    case 'h':
    default:
      fprintf(stderr,
              "Usage: %s --file <input_file> --iter <iterations> --mode "
              "<normal|mixed|opt>\n",
              argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (strlen(input_file) == 0) {
    fprintf(stderr, "Input file is required. Use --file <input_file>\n");
    return EXIT_FAILURE;
  }

  float *data = read_binary_floats(input_file, &nbEle);
  if (!data) {
    fprintf(stderr, "Failed to read input file: %s\n", input_file);
    return EXIT_FAILURE;
  }

  float *result = (float *)malloc(nbEle * sizeof(float));
  if (!result) {
    fprintf(stderr, "Failed to allocate memory for result buffer\n");
    free(data);
    return EXIT_FAILURE;
  }

  switch (mode) {
  case 0: // normal
  {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0, device_per_node = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    float *d_sbuf = nullptr, *d_rbuf = nullptr;

    cudaGetDeviceCount(&device_per_node);
    cudaSetDevice(rank % device_per_node);
    cudaMalloc((void **)&d_sbuf, nbEle * sizeof(float));
    cudaMemcpy(d_sbuf, data, nbEle * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_rbuf, nbEle * sizeof(float));

    double MPI_timer = 0.0;

    /*** MY_ALLREDUCE ***/
    for (int i = 0; i < iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_timer -= MPI_Wtime();
      allreduce_ring_comprs_hom_sum_F(d_sbuf, d_rbuf, nbEle, MPI_COMM_WORLD,
                                      eb);
      MPI_timer += MPI_Wtime();
    }
    double latency = MPI_timer / iterations;
    double min_time = 0.0, max_time = 0.0, avg_time = 0.0;
    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_time = avg_time / size;
    if (rank == 0) {
      printf("Compressed allreduce Min time: %f seconds\n", min_time);
      printf("Compressed allreduce Max time: %f seconds\n", max_time);
      printf("Compressed allreduce Avg time: %f seconds\n", avg_time);
      printf("Compressed allreduce Iterations: %d\n", iterations);
      printf("Compressed allreduce Count: %zu\n", nbEle);
    }

    /***  MPI_ALLREDUCE ***/
    MPI_timer = 0.0, latency = 0.0;
    max_time = 0.0, min_time = 0.0, avg_time = 0.0;
    for (int i = 0; i < iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_timer -= MPI_Wtime();
      allreduce_ring_gpu(d_sbuf, d_rbuf, nbEle, MPI_COMM_WORLD);
      MPI_timer += MPI_Wtime();
    }
    latency = MPI_timer / iterations;
    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_time = avg_time / size;
    if (rank == 0) {
      printf("MPI_Allreduce Min time: %f seconds\n", min_time);
      printf("MPI_Allreduce Max time: %f seconds\n", max_time);
      printf("MPI_Allreduce Avg time: %f seconds\n", avg_time);
    }
    cudaFree(d_sbuf);
    cudaFree(d_rbuf);
    MPI_Finalize();
    break;
  }
  case 1: // mixed
  {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0, device_per_node = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cudaGetDeviceCount(&device_per_node);
    cudaSetDevice(rank % device_per_node);
    float *d_sbuf = nullptr, *d_rbuf = nullptr;
    cudaMalloc((void **)&d_sbuf, nbEle * sizeof(float));
    cudaMemcpy(d_sbuf, data, nbEle * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_rbuf, nbEle * sizeof(float));
    double MPI_timer = 0.0;

    /*** MY_ALLREDUCE ***/
    for (int i = 0; i < iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_timer -= MPI_Wtime();
      mixed_compressed_allreduce(d_sbuf, d_rbuf, nbEle, MPI_COMM_WORLD, eb);
      MPI_timer += MPI_Wtime();
    }
    double latency = MPI_timer / iterations;
    double min_time = 0.0, max_time = 0.0, avg_time = 0.0;
    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_time = avg_time / size;
    if (rank == 0) {
      printf("Mixed compressed allreduce Min time: %f seconds\n", min_time);
      printf("Mixed compressed allreduce Max time: %f seconds\n", max_time);
      printf("Mixed compressed allreduce Avg time: %f seconds\n", avg_time);
      printf("Mixed compressed allreduce Iterations: %d\n", iterations);
      printf("Mixed compressed allreduce Count: %zu\n", nbEle);
    }

    /***  MPI_ALLREDUCE ***/
    MPI_timer = 0.0, latency = 0.0;
    max_time = 0.0, min_time = 0.0, avg_time = 0.0;
    for (int i = 0; i < iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_timer -= MPI_Wtime();
      allreduce_ring_gpu(d_sbuf, d_rbuf, nbEle, MPI_COMM_WORLD);
      MPI_timer += MPI_Wtime();
    }
    latency = MPI_timer / iterations;
    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_time = avg_time / size;
    if (rank == 0) {
      printf("MPI_Allreduce Min time: %f seconds\n", min_time);
      printf("MPI_Allreduce Max time: %f seconds\n", max_time);
      printf("MPI_Allreduce Avg time: %f seconds\n", avg_time);
    }
    CUDA_CHECK(cudaFree(d_sbuf));
    CUDA_CHECK(cudaFree(d_rbuf));
    MPI_Finalize();
    break;
  }
  case 2: // opt
  {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0, device_per_node = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    float *d_sbuf = nullptr, *d_rbuf = nullptr;
    cudaGetDeviceCount(&device_per_node);
    cudaSetDevice(rank % device_per_node);
    CUDA_CHECK(cudaMalloc((void **)&d_sbuf, nbEle * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sbuf, data, nbEle * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_rbuf, nbEle * sizeof(float)));
    double MPI_timer = 0.0;

    /*** MY_ALLREDUCE ***/

    for (int i = 0; i < iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_timer -= MPI_Wtime();
      allreduce_ring_comprs_hom_sum_F_opt(d_sbuf, d_rbuf, nbEle, MPI_COMM_WORLD,
                                          eb);
      MPI_timer += MPI_Wtime();
    }
    double latency = MPI_timer / iterations;
    double min_time = 0.0, max_time = 0.0, avg_time = 0.0;
    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_time = avg_time / size;
    if (rank == 0) {
      printf("Compressed allreduce Min time: %f seconds\n", min_time);
      printf("Compressed allreduce Max time: %f seconds\n", max_time);
      printf("Compressed allreduce Avg time: %f seconds\n", avg_time);
      printf("Compressed allreduce Iterations: %d\n", iterations);
      printf("Compressed allreduce Count: %zu\n", nbEle);
    }

    /***  MPI_ALLREDUCE ***/

    MPI_timer = 0.0, latency = 0.0;
    max_time = 0.0, min_time = 0.0, avg_time = 0.0;
    for (int i = 0; i < iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_timer -= MPI_Wtime();
      allreduce_ring_gpu(d_sbuf, d_rbuf, nbEle, MPI_COMM_WORLD);
      MPI_timer += MPI_Wtime();
    }
    latency = MPI_timer / iterations;
    MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    avg_time = avg_time / size;
    if (rank == 0) {
      printf("MPI_Allreduce Min time: %f seconds\n", min_time);
      printf("MPI_Allreduce Max time: %f seconds\n", max_time);
      printf("MPI_Allreduce Avg time: %f seconds\n", avg_time);
    }
    CUDA_CHECK(cudaFree(d_sbuf));
    CUDA_CHECK(cudaFree(d_rbuf));
    MPI_Finalize();
    break;
  }
  default:
    fprintf(stderr,
            "Invalid mode specified. Use --mode normal or --mode mixed.\n");
    free(data);
    free(result);
    return EXIT_FAILURE;
  }
  free(data);
  free(result);
  return EXIT_SUCCESS;
}
