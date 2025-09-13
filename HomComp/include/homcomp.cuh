#pragma once
__global__ void
kernel_quant_prediction(const float *const __restrict__ localData,
                        int *const __restrict__ quantPredData, const float eb,
                        const size_t nbEle);

__global__ void
kernel_homomophic_sum(const unsigned char *const __restrict__ CmpDataIn,
                      volatile unsigned int *const __restrict__ CmpOffsetIn,
                      unsigned char *const __restrict__ CmpDataOut,
                      volatile unsigned int *const __restrict__ locOffsetOut,
                      volatile unsigned int *const __restrict__ CmpOffsetOut,
                      volatile unsigned int *const __restrict__ locOffsetIn,
                      volatile int *const __restrict__ flag,
                      volatile int *const __restrict__ flag_cmp,
                      int *const __restrict__ predQuant, const float eb,
                      const size_t nbEle, const size_t cmpSize);

__global__ void
kernel_homomophic_sum_F(const unsigned char *const __restrict__ CmpDataIn,
                        volatile unsigned int *const __restrict__ CmpOffsetIn,
                        unsigned char *const __restrict__ CmpDataOut,
                        volatile unsigned int *const __restrict__ locOffsetOut,
                        volatile unsigned int *const __restrict__ CmpOffsetOut,
                        volatile unsigned int *const __restrict__ locOffsetIn,
                        volatile int *const __restrict__ flag,
                        volatile int *const __restrict__ flag_cmp,
                        float *const __restrict__ localChunk, const float eb,
                        const size_t nbEle, const size_t cmpSize);