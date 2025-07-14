#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/GSZ_timer.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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

void write_datai(const char *filename, int *data, size_t dim) {
  FILE *file = fopen(filename, "w");
  if (!file) {
    perror("Err");
    return;
  }

  for (size_t i = 0; i < dim; i++) {
    fprintf(file, "%d\n", data[i]);
  }

  fclose(file);
}

int main() {
  size_t nbEle;
  float *vec = read_data("randomwalk.in", &nbEle);
  float *vec_local = read_data("randomwalk.in", &nbEle);
  float eb = 1e-4;
  unsigned char *cmpBytes = NULL;
  float *decData;

  cmpBytes = (unsigned char *)malloc(nbEle * sizeof(float));
  decData = (float *)malloc(nbEle * sizeof(float));

  float *d_decData;
  float *d_vec;
  unsigned char *d_cmpBytes;
  size_t pad_nbEle = (nbEle + 32768 - 1) / 32768 * 32768;
  float max_val = vec[0];
  float min_val = vec[0];
  for (size_t i = 0; i < nbEle; i++) {
    if (vec[i] > max_val)
      max_val = vec[i];
    else if (vec[i] < min_val)
      min_val = vec[i];
  }
  eb = eb * (max_val - min_val);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMalloc((void **)&d_vec, pad_nbEle * sizeof(float));
  cudaMemcpy(d_vec, vec, pad_nbEle * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_decData, pad_nbEle * sizeof(float));
  cudaMalloc((void **)&d_cmpBytes, pad_nbEle * sizeof(float));

  size_t cmpSize;

  // Homomorphic compression simulation:
  /* 1. Compress the data (we've to send data to another device, but for the
   purpose of testing, we can do all operation on the same device)
    2. Apply prediction + quantization to data local to the device
    3. Apply homomoprhic sum kernel
    4. Decompress the data and write it to the output file
  */
  GSZ_compress_deviceptr_outlier(d_vec, d_cmpBytes, nbEle, &cmpSize, eb,
                                 stream);
  float *d_localData;
  int *d_quantLocOut;

  cudaMalloc((void **)&d_localData, pad_nbEle * sizeof(float));
  cudaMemcpy(d_localData, vec_local, pad_nbEle * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_quantLocOut, pad_nbEle * sizeof(int));

  int bsize = dec_tblock_size;
  int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
  dim3 grid(gsize);
  dim3 block(bsize);
  int *d_quantLocOut2;
  cudaMalloc((void **)&d_quantLocOut2, pad_nbEle * sizeof(int));
  unsigned char *d_cmpBytesOut;
  cudaMalloc((void **)&d_cmpBytesOut, pad_nbEle * sizeof(float));
  size_t cmpSize2;
  unsigned char *d_cmpBytesOut2;
  cudaMalloc((void **)&d_cmpBytesOut2, pad_nbEle * sizeof(float));
  homomorphic_sum_F(d_cmpBytes, d_localData, d_cmpBytesOut, nbEle, eb,
                    &cmpSize);
  homomorphic_sum_F(d_cmpBytesOut, d_localData, d_cmpBytesOut2, nbEle, eb,
                    &cmpSize, 0);
  homomorphic_sum_F(d_cmpBytesOut2, d_localData, d_cmpBytesOut, nbEle, eb,
                    &cmpSize, 0);
  homomorphic_sum_F(d_cmpBytesOut, d_localData, d_cmpBytesOut2, nbEle, eb,
                    &cmpSize, 0);
  homomorphic_sum_F(d_cmpBytesOut2, d_localData, d_cmpBytesOut, nbEle, eb,
                    &cmpSize, 0);
  homomorphic_sum_F(d_cmpBytesOut, d_localData, d_cmpBytesOut2, nbEle, eb,
                    &cmpSize, 0);
  homomorphic_sum_F(d_cmpBytesOut2, d_localData, d_cmpBytesOut, nbEle, eb,
                    &cmpSize, 0);
  GSZ_decompress_deviceptr_outlier(d_decData, d_cmpBytesOut, nbEle, cmpSize2,
                                   eb);

  cudaMemcpy(decData, d_decData, nbEle * sizeof(float), cudaMemcpyDeviceToHost);

  write_dataf("output", decData, nbEle);

  int not_bound = 0;
  for (size_t i = 0; i < nbEle; i += 1) {
    if (fabs(vec[i] * 3 - decData[i]) > eb * 3.3) {
      not_bound++;
    }
  }
  if (!not_bound)
    printf("\033[0;32mPass error check!\033[0m\n");
  else
    printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n",
           not_bound);
  cudaFree(d_cmpBytes);
  cudaFree(d_cmpBytesOut);
  cudaFree(d_decData);
  cudaFree(d_vec);
  cudaFree(d_localData);
  cudaFree(d_quantLocOut);
  free(cmpBytes);
  free(vec);
  free(vec_local);
  free(decData);
  cudaStreamDestroy(stream);
  return 0;
}