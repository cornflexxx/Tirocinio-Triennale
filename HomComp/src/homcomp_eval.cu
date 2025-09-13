#include "../include/GSZ.h"
#include "../include/GSZ_entry.h"
#include "../include/GSZ_timer.h"
#include "../include/homcomp.cuh"
#include "../include/readFile.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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
#define ITERATIONS
double totalCost = 0;
struct timeval costStart;
struct timeval endTime;
struct timeval startTime;

__global__ void sum2arrays(float *__restrict__ a, const float *__restrict__ b,
                           size_t count) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < count; i += stride) {
    a[i] += b[i];
  }
}

void cost_start() {
  totalCost = 0;
  gettimeofday(&costStart, NULL);
}

void cost_end() {
  double elapsed;
  struct timeval costEnd;
  gettimeofday(&costEnd, NULL);
  elapsed = ((costEnd.tv_sec * 1000000 + costEnd.tv_usec) -
             (costStart.tv_sec * 1000000 + costStart.tv_usec)) /
            1000000.0;
  totalCost += elapsed;
}

void error_evaluation(float *oriData, float *decData, size_t nbEle,
                      size_t cmpSize) {
  size_t i = 0;
  float Max = 0, Min = 0, diffMax = 0;
  Max = oriData[0];
  Min = oriData[0];
  diffMax = fabs(decData[0] - oriData[0]);
  double sum1 = 0, sum2 = 0;
  for (i = 0; i < nbEle; i++) {
    sum1 += oriData[i];
    sum2 += decData[i];
  }
  double mean1 = sum1 / nbEle;
  double mean2 = sum2 / nbEle;

  double sum3 = 0, sum4 = 0;
  double sum = 0, prodSum = 0, relerr = 0;

  double maxpw_relerr = 0;
  for (i = 0; i < nbEle; i++) {
    if (Max < oriData[i])
      Max = oriData[i];
    if (Min > oriData[i])
      Min = oriData[i];

    float err = fabs(decData[i] - oriData[i]);
    if (oriData[i] != 0) {
      if (fabs(oriData[i]) > 1)
        relerr = err / oriData[i];
      else
        relerr = err;
      if (maxpw_relerr < relerr)
        maxpw_relerr = relerr;
    }

    if (diffMax < err)
      diffMax = err;
    prodSum += (oriData[i] - mean1) * (decData[i] - mean2);
    sum3 += (oriData[i] - mean1) * (oriData[i] - mean1);
    sum4 += (decData[i] - mean2) * (decData[i] - mean2);
    sum += err * err;
  }
  double std1 = sqrt(sum3 / nbEle);
  double std2 = sqrt(sum4 / nbEle);
  double ee = prodSum / nbEle;
  double acEff = ee / std1 / std2;

  double mse = sum / nbEle;
  double range = Max - Min;
  double psnr = 20 * log10(range) - 10 * log10(mse);
  double nrmse = sqrt(mse) / range;

  double compressionRatio = 1.0 * nbEle * sizeof(float) / cmpSize;

  printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
  printf("Max absolute error = %.10f\n", diffMax);
  printf("Max relative error = %f\n", diffMax / (Max - Min));
  printf("Max pw relative error = %f\n", maxpw_relerr);
  printf("PSNR = %.3f, NRMSE= %.5G\n", psnr, nrmse);
  printf("acEff=%f\n", acEff);
  printf("Compression Ratio = %f\n", compressionRatio);
}

int main(int argc, char **argv) {
  size_t nbEle;
  float *vec = read_binary_floats(
      "datasets/normal_smooth-True_idx-000_953.67MB.bin", &nbEle);
  float *vec_local = read_binary_floats(
      "datasets/normal_smooth-True_idx-000_953.67MB.bin", &nbEle);
  float eb = 1e-4;
  int rel = 0;
  unsigned char *cmpBytes = NULL;
  float *decData;
  if (argc > 1)
    eb = atoi(argv[1]);
  if (argc > 2)
    rel = atoi((const char *)argv[2]);
  cmpBytes = (unsigned char *)malloc(nbEle * sizeof(float));
  decData = (float *)malloc(nbEle * sizeof(float));

  float *d_decData;
  float *d_vec;
  unsigned char *d_cmpBytes;
  size_t pad_nbEle = (nbEle + 32768 - 1) / 32768 * 32768;
  if (rel) {
    float max_val = vec[0];
    float min_val = vec[0];
    for (size_t i = 0; i < nbEle; i++) {
      if (vec[i] > max_val)
        max_val = vec[i];
      else if (vec[i] < min_val)
        min_val = vec[i];
    }
    eb = eb * (max_val - min_val);
  } // REL

  size_t cmpSize, cmpSize2;
  float *d_localData;
  unsigned char *d_cmpBytesOut;
  unsigned char *d_cmpBytesOut2;
  double hom_cost, cmpt_cos, compr_cost, decomp_cost;
  cudaMalloc((void **)&d_vec, pad_nbEle * sizeof(float));
  cudaMemcpy(d_vec, vec, nbEle * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_decData, pad_nbEle * sizeof(float));
  cudaMalloc((void **)&d_cmpBytes, pad_nbEle * sizeof(float));
  cudaMalloc((void **)&d_cmpBytesOut, pad_nbEle * sizeof(float));
  cudaMalloc((void **)&d_localData, pad_nbEle * sizeof(float));
  cudaMemcpy(d_localData, vec_local, nbEle * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_cmpBytesOut2, pad_nbEle * sizeof(float));
  GSZ_compress_deviceptr_outlier(d_vec, d_cmpBytes, pad_nbEle, &cmpSize, eb);
  size_t bsize = dec_tblock_size;
  size_t gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);

  dim3 grid(gsize);
  dim3 block(bsize);

  cost_start();
  GSZ_decompress_deviceptr_outlier(d_decData, d_cmpBytes, nbEle, cmpSize, eb);
  cost_end();
  decomp_cost = totalCost;
  cost_start();
  sum2arrays<<<grid, block>>>(d_decData, d_decData, nbEle);
  cost_end();
  cmpt_cos = totalCost;

  cost_start();
  GSZ_compress_deviceptr_outlier(d_decData, d_cmpBytes, nbEle, &cmpSize, eb);
  cost_end();

  compr_cost = totalCost;
  double normal_cost = cmpt_cos + compr_cost + decomp_cost;

  GSZ_compress_deviceptr_outlier(d_vec, d_cmpBytes, nbEle, &cmpSize, eb);
  cost_start();
  homomorphic_sum_F(d_cmpBytes, d_localData, d_cmpBytesOut, pad_nbEle, eb,
                    &cmpSize2, cmpSize);
  cost_end();
  hom_cost = totalCost;
  GSZ_decompress_deviceptr_outlier(d_decData, d_cmpBytesOut, nbEle, cmpSize2,
                                   eb);
  cudaMemcpy(decData, d_decData, nbEle * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < nbEle; i++)
    vec[i] *= 2;
  error_evaluation(vec, decData, nbEle, cmpSize2);
  write_dataf("output.txt", decData, nbEle);

  printf("Traditional DOC workflow (decompression+operation+compression) "
         "performance: time: %f s, throughput: %f GBps\n",
         normal_cost, nbEle * sizeof(float) / normal_cost / 1000 / 1000 / 1000);
  printf("HomComp_F performance: time: %f s, throughput: %f GBps\n", hom_cost,
         nbEle * sizeof(float) / hom_cost / 1000 / 1000 / 1000);
  printf("HomComp_F speedup: %0.2fX\n", normal_cost / hom_cost);
  cudaFree(d_cmpBytes);
  cudaFree(d_cmpBytesOut);
  cudaFree(d_decData);
  cudaFree(d_vec);
  cudaFree(d_localData);
  free(cmpBytes);
  free(vec);
  free(vec_local);
  free(decData);
  return 0;
}