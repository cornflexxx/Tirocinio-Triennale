

#include "../include/GSZ.h"
#include "../include/comprs_test.cuh"
#include <cstddef>
#include <cstdio>
#include <stdio.h>

__device__ inline int quantization(float data, float recipPrecision) {
  int result;
  asm("{\n\t"
      ".reg .f32 dataRecip;\n\t"
      ".reg .f32 temp1;\n\t"
      ".reg .s32 s;\n\t"
      ".reg .pred p;\n\t"
      "mul.f32 dataRecip, %1, %2;\n\t"      // dataRecip = data * recipPrecision
      "setp.ge.f32 p, dataRecip, -0.5;\n\t" // Set predicate if dataRecip >=
                                            // -0.5
      "selp.s32 s, 0, 1, p;\n\t"            // s = 0 if p is true, else s = 1
      "add.f32 temp1, dataRecip, 0.5;\n\t"  // temp1 = dataRecip + 0.5
      "cvt.rzi.s32.f32 %0, temp1;\n\t" // Convert to int with round towards zero
                                       // and store to result
      "sub.s32 %0, %0, s;\n\t"         // result = result - s
      "}"
      : "=r"(result)
      : "f"(data), "f"(recipPrecision));
  return result;
}

__device__ inline int get_bit_num(unsigned int x) {
  int leading_zeros;
  asm("clz.b32 %0, %1;" : "=r"(leading_zeros) : "r"(x));
  return 32 - leading_zeros;
}

// ______________________________________________________________________
__global__ void GSZ_compress_kernel_outlier_vec(
    const float *const __restrict__ oriData,
    unsigned char *const __restrict__ CmpDataIn,
    volatile unsigned int *const __restrict__ CmpOffsetIn,
    volatile unsigned int *const __restrict__ locOffsetIn,
    volatile int *const __restrict__ flag, const float eb, const size_t nbEle) {
  __shared__ unsigned int excl_sum;
  __shared__ unsigned int base_idx;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid * blockDim.x + tid;
  const int lane = idx & 0x1f;
  const int warp = idx >> 5;
  const int block_num = cmp_chunk >> 5;
  const int rate_ofs = (nbEle + cmp_tblock_size * cmp_chunk - 1) /
                       (cmp_tblock_size * cmp_chunk) *
                       (cmp_tblock_size * cmp_chunk) / 32;
  const float recipPrecision = 0.5f / eb;

  int base_start_idx;
  int base_block_start_idx, base_block_end_idx;
  int quant_chunk_idx;
  int block_idx;
  int currQuant, lorenQuant, prevQuant;
  int absQuant[cmp_chunk];
  unsigned int sign_flag[block_num];
  int sign_ofs;
  int fixed_rate[block_num];
  unsigned int thread_ofs = 0;
  uchar4 tmp_char;

  // Prequantization + Lorenzo Prediction + Fixed-length encoding + store
  // fixed-length to global memory.
  base_start_idx = warp * cmp_chunk * 32;
  for (int j = 0; j < block_num; j++) {
    // Block initilization.
    base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
    base_block_end_idx = base_block_start_idx + 32;
    sign_flag[j] = 0;
    fixed_rate[j] = 0;
    block_idx = base_block_start_idx / 32;
    prevQuant = 0;
    int maxQuant = 0;
    int maxQuan2 = 0;
    int outlier = 0;

    int i = base_block_start_idx;
    quant_chunk_idx = j * 32 + i % 32;

    float4 tmp_buffer =
        reinterpret_cast<const float4 *>(oriData)[base_block_start_idx];
    int4 tmp_quant1;

    tmp_quant1.x = quantization(tmp_buffer.x, recipPrecision);
    tmp_quant1.y = quantization(tmp_buffer.y, recipPrecision);
    tmp_quant1.z = quantization(tmp_buffer.z, recipPrecision);
    tmp_quant1.w = quantization(tmp_buffer.w, recipPrecision);

    prevQuant = tmp_quant1.x;
    sign_ofs = i % 32;
    sign_flag[j] |= (prevQuant < 0) << (31 - sign_ofs);
    outlier = absQuant[quant_chunk_idx];
    absQuant[quant_chunk_idx] = abs(prevQuant);
    maxQuant = abs(prevQuant); // absQuant[quant_chunk_idx]

    currQuant = tmp_quant1.y;
    lorenQuant = currQuant - prevQuant;
    prevQuant = currQuant;
    sign_flag[j] |= (lorenQuant < 0) << (31 - (sign_ofs + 1));
    absQuant[quant_chunk_idx] = abs(lorenQuant);
    maxQuant = max(maxQuant, absQuant[quant_chunk_idx]);
    maxQuan2 = max(maxQuan2, absQuant[quant_chunk_idx]);

    currQuant = tmp_quant1.z;
    lorenQuant = currQuant - prevQuant;
    prevQuant = currQuant;
    sign_flag[j] |= (lorenQuant < 0) << (31 - (sign_ofs + 2));
    absQuant[quant_chunk_idx] = abs(lorenQuant);
    maxQuant = max(maxQuant, absQuant[quant_chunk_idx]);
    maxQuan2 = max(maxQuan2, absQuant[quant_chunk_idx]);

    currQuant = tmp_quant1.w;
    lorenQuant = currQuant - prevQuant;
    prevQuant = currQuant;
    sign_flag[j] |= (lorenQuant < 0) << (31 - (sign_ofs + 3));
    absQuant[quant_chunk_idx] = abs(lorenQuant);
    maxQuant = max(maxQuant, absQuant[quant_chunk_idx]);
    maxQuan2 = max(maxQuan2, absQuant[quant_chunk_idx]);

    int4 tmp_quant;
    int4 tmp_lorenPred;
// Operation for each block
#pragma unroll 8
    for (i += 4; i < base_block_end_idx; i += 4) {
      // Read data from global memory.
      tmp_buffer = reinterpret_cast<const float4 *>(oriData)[i / 4];
      quant_chunk_idx = j * 32 + i % 32;

      /* NOTE : quantization4 quindi vettorizzata -> esistono istruzioni PTX
       ottimizzate per questo caso (?) */
      tmp_quant.x = quantization(tmp_buffer.x, recipPrecision);
      tmp_quant.y = quantization(tmp_buffer.y, recipPrecision);
      tmp_quant.z = quantization(tmp_buffer.z, recipPrecision);
      tmp_quant.w = quantization(tmp_buffer.w, recipPrecision);

      tmp_lorenPred.x = tmp_quant.x - tmp_quant1.x;
      tmp_lorenPred.y = tmp_quant.y - tmp_quant1.y;
      tmp_lorenPred.z = tmp_quant.z - tmp_quant1.z;
      tmp_lorenPred.w = tmp_quant.w - tmp_quant1.w;

      tmp_quant1 = tmp_quant;

      sign_flag[j] |= ((tmp_lorenPred.x < 0) << 3 | (tmp_lorenPred.y < 0) << 2 |
                       (tmp_lorenPred.z < 0) << 1 | (tmp_lorenPred.w < 0))
                      << (31 - (i % 32));

      absQuant[quant_chunk_idx] = abs(tmp_lorenPred.x);
      absQuant[quant_chunk_idx + 1] = abs(tmp_lorenPred.y);
      absQuant[quant_chunk_idx + 2] = abs(tmp_lorenPred.z);
      absQuant[quant_chunk_idx + 3] = abs(tmp_lorenPred.w);

      int tmp_quant = max(max(tmp_lorenPred.x, tmp_lorenPred.y),
                          max(tmp_lorenPred.z, tmp_lorenPred.w));
      maxQuant = max(maxQuant, tmp_quant);
      maxQuan2 = max(maxQuan2, tmp_quant);
      // reinterpret_cast<int4 *>(absQuant)[i / 4] = tmp_lorenPred;
    }

    // Outlier fixed-length encoding selection.
    int fr1 = get_bit_num(maxQuant);
    int fr2 = get_bit_num(maxQuan2);
    outlier = (get_bit_num(outlier) + 7) / 8;
    int temp_rate = 0;
    int temp_ofs1 = fr1 ? 4 + fr1 * 4 : 0;
    int temp_ofs2 = fr2 ? 4 + fr2 * 4 + outlier : 4 + outlier;
    if (temp_ofs1 <= temp_ofs2) {
      thread_ofs += temp_ofs1;
      temp_rate = fr1;
    } else {
      thread_ofs += temp_ofs2;
      temp_rate = fr2 | 0x80 | ((outlier - 1) << 5);
    }

    // Record block info and write block fixed rate to compressed data.
    fixed_rate[j] = temp_rate;
    CmpDataIn[block_idx] = (unsigned char)fixed_rate[j];
    __syncthreads();
  }

// Warp-level prefix-sum (inclusive), also thread-block-level.
#pragma unroll 5
  for (int i = 1; i < 32; i <<= 1) {
    int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
    if (lane >= i)
      thread_ofs += tmp;
  }
  __syncthreads();

  // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
  if (lane == 31) {
    locOffsetIn[warp + 1] = thread_ofs;
    __threadfence();
    if (warp == 0) {
      flag[0] = 2;
      __threadfence();
      flag[1] = 1;
      __threadfence();
    } else {
      flag[warp + 1] = 1;
      __threadfence();
    }
  }
  __syncthreads();

  // Global-level prefix-sum (exclusive).
  if (warp > 0) {
    if (!lane) {
      // Decoupled look-back
      int lookback = warp;
      int loc_excl_sum = 0;
      while (lookback > 0) {
        int status;
        // Local sum not end.
        do {
          status = flag[lookback];
          __threadfence();
        } while (status == 0);
        // Lookback end.
        if (status == 2) {
          loc_excl_sum += CmpOffsetIn[lookback];
          __threadfence();
          break;
        }
        // Continues lookback.
        if (status == 1)
          loc_excl_sum += locOffsetIn[lookback];
        lookback--;
        __threadfence();
      }
      excl_sum = loc_excl_sum;
    }
    __syncthreads();
  }

  if (warp > 0) {
    // Update global flag.
    if (!lane)
      CmpOffsetIn[warp] = excl_sum;
    __threadfence();
    if (!lane)
      flag[warp] = 2;
    __threadfence();
  }
  __syncthreads();

  // Assigning compression bytes by given prefix-sum results.
  if (!lane)
    base_idx = excl_sum + rate_ofs;
  __syncthreads();

  // Bit shuffle for each index, also storing data to global memory.
  unsigned int base_cmp_byte_ofs = base_idx;
  unsigned int cmp_byte_ofs;
  unsigned int tmp_byte_ofs = 0;
  unsigned int cur_byte_ofs = 0;
  for (int j = 0; j < block_num; j++) {
    // Initialization, guiding encoding process.
    int encoding_selection = fixed_rate[j] >> 7;
    int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
    fixed_rate[j] &= 0x1f;
    int chunk_idx_start = j * 32;

    // Restore index for j-th iteration.
    if (!encoding_selection)
      tmp_byte_ofs = (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
    else
      tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
      int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
      if (lane >= i)
        tmp_byte_ofs += tmp;
    }
    unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
    if (!lane)
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
    else
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

    // If outlier encoding, storing outliers here.
    if (encoding_selection) {
      for (int i = 0; i < outlier_byte_num; i++) {
        CmpDataIn[cmp_byte_ofs++] =
            (unsigned char)(absQuant[chunk_idx_start] & 0xff);
        absQuant[chunk_idx_start] >>= 8;
      }

      // Corner case: all data points except outliers are 0.
      if (!fixed_rate[j]) {
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);
        CmpDataIn[cmp_byte_ofs++] = 0xff & sign_flag[j];
      }
    }

    // Operation for each block, if zero block then do nothing.
    if (fixed_rate[j]) {
      // Padding vector operation for outlier encoding.
      int vec_ofs = cmp_byte_ofs % 4;
      if (vec_ofs == 0) {
        // Assign sign information for one block.
        tmp_char.x = 0xff & (sign_flag[j] >> 24);
        tmp_char.y = 0xff & (sign_flag[j] >> 16);
        tmp_char.z = 0xff & (sign_flag[j] >> 8);
        tmp_char.w = 0xff & sign_flag[j];
        reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate[j]; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.x = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.x = tmp_char.x |
                       (((absQuant[chunk_idx_start + 0] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 1] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 2] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 3] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 4] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 5] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 6] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 7] & mask) >> i) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.y = (((absQuant[chunk_idx_start + 8] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 9] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 10] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 11] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 12] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 13] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 14] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 15] & mask) >> i) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.z = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.w = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
          mask <<= 1;
        }
      } else if (vec_ofs == 1) {
        // Assign sign information for one block, padding part.
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & sign_flag[j];
        if (!encoding_selection)
          tmp_char.y = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.y = tmp_char.y | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        tmp_char.z = ((absQuant[chunk_idx_start + 8] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 9] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 10] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 11] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 12] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 13] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 14] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 15] & 1) << 0);
        tmp_char.w = ((absQuant[chunk_idx_start + 16] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 17] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 18] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 19] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 20] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 21] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 22] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 23] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.x = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.y = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.y =
              tmp_char.y |
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.z =
              (((absQuant[chunk_idx_start + 8] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 9] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 10] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 11] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 12] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 13] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 14] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 15] & mask) >> (i + 1)) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.w =
              (((absQuant[chunk_idx_start + 16] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 17] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 18] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 19] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 20] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 21] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 22] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 23] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataIn[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >> (fixed_rate[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >> (fixed_rate[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >> (fixed_rate[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >> (fixed_rate[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >> (fixed_rate[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >> (fixed_rate[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >> (fixed_rate[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >> (fixed_rate[j] - 1))
             << 0);
      } else if (vec_ofs == 2) {
        // Assign sign information for one block, padding part.
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & (sign_flag[j] >> 8);
        tmp_char.y = 0xff & sign_flag[j];
        if (!encoding_selection)
          tmp_char.z = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.z = tmp_char.z | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        tmp_char.w = ((absQuant[chunk_idx_start + 8] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 9] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 10] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 11] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 12] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 13] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 14] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 15] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.x = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.y = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.z = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.z =
              tmp_char.z |
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.w =
              (((absQuant[chunk_idx_start + 8] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 9] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 10] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 11] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 12] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 13] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 14] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 15] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataIn[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 16] & mask) >> (fixed_rate[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 17] & mask) >> (fixed_rate[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 18] & mask) >> (fixed_rate[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 19] & mask) >> (fixed_rate[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 20] & mask) >> (fixed_rate[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 21] & mask) >> (fixed_rate[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 22] & mask) >> (fixed_rate[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 23] & mask) >> (fixed_rate[j] - 1))
             << 0);
        CmpDataIn[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >> (fixed_rate[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >> (fixed_rate[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >> (fixed_rate[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >> (fixed_rate[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >> (fixed_rate[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >> (fixed_rate[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >> (fixed_rate[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >> (fixed_rate[j] - 1))
             << 0);
      } else {
        // Assign sign information for one block, padding part.
        CmpDataIn[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & (sign_flag[j] >> 16);
        tmp_char.y = 0xff & (sign_flag[j] >> 8);
        tmp_char.z = 0xff & sign_flag[j];
        if (!encoding_selection)
          tmp_char.w = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.w = tmp_char.w | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.x = (((absQuant[chunk_idx_start + 8] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 9] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 10] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 11] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 12] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 13] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 14] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 15] & mask) >> i) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.y = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.z = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.w = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.w =
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataIn[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 8] & mask) >> (fixed_rate[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 9] & mask) >> (fixed_rate[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 10] & mask) >> (fixed_rate[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 11] & mask) >> (fixed_rate[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 12] & mask) >> (fixed_rate[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 13] & mask) >> (fixed_rate[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 14] & mask) >> (fixed_rate[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 15] & mask) >> (fixed_rate[j] - 1))
             << 0);
        CmpDataIn[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 16] & mask) >> (fixed_rate[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 17] & mask) >> (fixed_rate[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 18] & mask) >> (fixed_rate[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 19] & mask) >> (fixed_rate[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 20] & mask) >> (fixed_rate[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 21] & mask) >> (fixed_rate[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 22] & mask) >> (fixed_rate[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 23] & mask) >> (fixed_rate[j] - 1))
             << 0);
        CmpDataIn[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >> (fixed_rate[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >> (fixed_rate[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >> (fixed_rate[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >> (fixed_rate[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >> (fixed_rate[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >> (fixed_rate[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >> (fixed_rate[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >> (fixed_rate[j] - 1))
             << 0);
      }
    }

    // Index updating across different iterations.
    cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
  }
}

__global__ void GSZ_decompress_kernel_outlier_vec(
    float *const __restrict__ decData,
    const unsigned char *const __restrict__ CmpDataIn,
    volatile unsigned int *const __restrict__ CmpOffsetIn,
    volatile unsigned int *const __restrict__ locOffsetIn,
    volatile int *const __restrict__ flag, const float eb, const size_t nbEle) {
  __shared__ unsigned int excl_sum;
  __shared__ unsigned int base_idx;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid * blockDim.x + tid;
  const int lane = idx & 0x1f;
  const int warp = idx >> 5;
  const int block_num = dec_chunk >> 5;
  const int rate_ofs = (nbEle + dec_tblock_size * dec_chunk - 1) /
                       (dec_tblock_size * dec_chunk) *
                       (dec_tblock_size * dec_chunk) / 32;

  int base_start_idx;
  int base_block_start_idx;
  int block_idx;
  int absQuant[32];
  int currQuant, lorenQuant, prevQuant;
  int sign_ofs;
  int fixed_rate[block_num];
  unsigned int thread_ofs = 0;
  uchar4 tmp_char;
  float4 dec_buffer;
  // Obtain fixed rate information for each block.
  for (int j = 0; j < block_num; j++) {
    block_idx = warp * dec_chunk + j * 32 + lane;
    fixed_rate[j] = (int)CmpDataIn[block_idx];

    // Encoding selection.
    int encoding_selection = fixed_rate[j] >> 7;
    int outlier = ((fixed_rate[j] & 0x60) >> 5) + 1;
    int temp_rate = fixed_rate[j] & 0x1f;
    if (!encoding_selection)
      thread_ofs += temp_rate ? (4 + temp_rate * 4) : 0;
    else
      thread_ofs += 4 + temp_rate * 4 + outlier;
    __syncthreads();
  }

// Warp-level prefix-sum (inclusive), also thread-block-level.
#pragma unroll 5
  for (int i = 1; i < 32; i <<= 1) {
    int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
    if (lane >= i)
      thread_ofs += tmp;
  }
  __syncthreads();

  // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
  if (lane == 31) {
    locOffsetIn[warp + 1] = thread_ofs;
    __threadfence();
    if (warp == 0) {
      flag[0] = 2;
      __threadfence();
      flag[1] = 1;
      __threadfence();
    } else {
      flag[warp + 1] = 1;
      __threadfence();
    }
  }
  __syncthreads();

  // Global-level prefix-sum (exclusive).
  if (warp > 0) {
    if (!lane) {
      // Decoupled look-back
      int lookback = warp;
      int loc_excl_sum = 0;
      while (lookback > 0) {
        int status;
        // Local sum not end.
        do {
          status = flag[lookback];
          __threadfence();
        } while (status == 0);
        // Lookback end.
        if (status == 2) {
          loc_excl_sum += CmpOffsetIn[lookback];
          __threadfence();
          break;
        }
        // Continues lookback.
        if (status == 1)
          loc_excl_sum += locOffsetIn[lookback];
        lookback--;
        __threadfence();
      }
      excl_sum = loc_excl_sum;
    }
    __syncthreads();
  }

  if (warp > 0) {
    // Update global flag.
    if (!lane)
      CmpOffsetIn[warp] = excl_sum;
    __threadfence();
    if (!lane)
      flag[warp] = 2;
    __threadfence();
  }
  __syncthreads();

  // Retrieving compression bytes and reconstruct decompression data.
  if (!lane)
    base_idx = excl_sum + rate_ofs;
  __syncthreads();

  // Restore bit-shuffle for each block.
  unsigned int base_cmp_byte_ofs = base_idx;
  unsigned int cmp_byte_ofs;
  unsigned int tmp_byte_ofs = 0;
  unsigned int cur_byte_ofs = 0;
  base_start_idx = warp * dec_chunk * 32;
  for (int j = 0; j < block_num; j++) {
    // Initialization, guiding encoding process.
    int encoding_selection = fixed_rate[j] >> 7;
    int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
    fixed_rate[j] &= 0x1f;
    int outlier_buffer = 0;
    base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
    unsigned int sign_flag = 0;

    // Restore index for j-th iteration.
    if (!encoding_selection)
      tmp_byte_ofs = (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
    else
      tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
      int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
      if (lane >= i)
        tmp_byte_ofs += tmp;
    }
    unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
    if (!lane)
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
    else
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

    // If outlier encoding, retrieve outliers here.
    if (encoding_selection) {
      for (int i = 0; i < outlier_byte_num; i++) {
        int buffer = CmpDataIn[cmp_byte_ofs++] << (8 * i);
        outlier_buffer |= buffer;
      }

      // Corner case: all data points except outliers are 0.
      if (!fixed_rate[j]) {
        sign_flag = (0xff000000 & (CmpDataIn[cmp_byte_ofs++] << 24)) |
                    (0x00ff0000 & (CmpDataIn[cmp_byte_ofs++] << 16)) |
                    (0x0000ff00 & (CmpDataIn[cmp_byte_ofs++] << 8)) |
                    (0x000000ff & CmpDataIn[cmp_byte_ofs++]);
        absQuant[0] = outlier_buffer;
        for (int i = 1; i < 32; i++)
          absQuant[i] = 0;
        // ------------------------------------------ Const block
        // Delorenzo and store data back to decompression data.
        int i = 0;
        prevQuant = 0;
        // For the .x element, reconstruct sign (absolute value), lorenzo
        // quantization, quantization, and original value.
        sign_ofs = i % 32;
        lorenQuant =
            sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
        currQuant = lorenQuant + prevQuant;
        prevQuant = currQuant;
        dec_buffer.x = currQuant * eb * 2;

        // For the .y element, reconstruct sign (absolute value), lorenzo
        // quantization, quantization, and original value.
        sign_ofs = (i + 1) % 32;
        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i + 1] * -1
                                                        : absQuant[i + 1];
        currQuant = lorenQuant + prevQuant;
        prevQuant = currQuant;
        dec_buffer.y = currQuant * eb * 2;

        // For the .z element, reconstruct sign (absolute value), lorenzo
        // quantization, quantization, and original value.
        sign_ofs = (i + 2) % 32;
        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i + 2] * -1
                                                        : absQuant[i + 2];
        currQuant = lorenQuant + prevQuant;
        prevQuant = currQuant;
        dec_buffer.z = currQuant * eb * 2;

        // For the .w element, reconstruct sign (absolute value), lorenzo
        // quantization, quantization, and original value.
        sign_ofs = (i + 3) % 32;
        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i + 3] * -1
                                                        : absQuant[i + 3];
        currQuant = lorenQuant + prevQuant;
        prevQuant = currQuant;
        dec_buffer.w = currQuant * eb * 2;
        // Delorenzo and store data back to decompression data.
        reinterpret_cast<float4 *>(decData)[(base_block_start_idx + i) / 4] =
            dec_buffer;

        float4 dec_buffer_prec = dec_buffer;
#pragma unroll 8
        for (i = 4; i < 32; i += 4) {

          sign_ofs = i % 32;

          dec_buffer.x = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1
                                                            : absQuant[i];
          dec_buffer.y = sign_flag & (1 << (31 - (sign_ofs + 1)))
                             ? absQuant[i + 1] * -1
                             : absQuant[i + 1];
          dec_buffer.z = sign_flag & (1 << (31 - (sign_ofs + 2)))
                             ? absQuant[i + 2] * -1
                             : absQuant[i + 2];
          dec_buffer.w = sign_flag & (1 << (31 - (sign_ofs + 3)))
                             ? absQuant[i + 3] * -1
                             : absQuant[i + 3];

          dec_buffer.x = (dec_buffer.x + dec_buffer_prec.x);
          dec_buffer.y = (dec_buffer.y + dec_buffer_prec.y);
          dec_buffer.z = (dec_buffer.z + dec_buffer_prec.z);
          dec_buffer.w = (dec_buffer.w + dec_buffer_prec.w);

          dec_buffer_prec = dec_buffer;

          dec_buffer.x = dec_buffer.x * eb * 2;
          dec_buffer.y = dec_buffer.y * eb * 2;
          dec_buffer.z = dec_buffer.z * eb * 2;
          dec_buffer.w = dec_buffer.w * eb * 2;

          reinterpret_cast<float4 *>(decData)[(base_block_start_idx + i) / 4] =
              dec_buffer;
          // Read data from global variable via a vectorized pattern.
        }
      }
    }

    // Operation for each block, if zero block then do nothing.
    if (fixed_rate[j]) {
      // Padding vector operation for reverse outlier encoding.
      int vec_ofs = cmp_byte_ofs % 4;
      if (vec_ofs == 0) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[0] = outlier_buffer;

        // Retrieve sign information for one block.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                    (0x00ff0000 & (tmp_char.y << 16)) |
                    (0x0000ff00 & (tmp_char.z << 8)) |
                    (0x000000ff & tmp_char.w);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j]; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[0] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
          absQuant[17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
          absQuant[18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
          absQuant[19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
          absQuant[20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
          absQuant[21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
          absQuant[22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
          absQuant[23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
          absQuant[25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
          absQuant[26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
          absQuant[27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
          absQuant[28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
          absQuant[29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
          absQuant[30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
          absQuant[31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
        }
      } else if (vec_ofs == 1) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[0] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24) |
                    (0x00ff0000 & CmpDataIn[cmp_byte_ofs++] << 16) |
                    (0x0000ff00 & CmpDataIn[cmp_byte_ofs++] << 8);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x000000ff & tmp_char.x);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[0] |= ((tmp_char.y >> 7) & 0x00000001);
        absQuant[1] |= ((tmp_char.y >> 6) & 0x00000001);
        absQuant[2] |= ((tmp_char.y >> 5) & 0x00000001);
        absQuant[3] |= ((tmp_char.y >> 4) & 0x00000001);
        absQuant[4] |= ((tmp_char.y >> 3) & 0x00000001);
        absQuant[5] |= ((tmp_char.y >> 2) & 0x00000001);
        absQuant[6] |= ((tmp_char.y >> 1) & 0x00000001);
        absQuant[7] |= ((tmp_char.y >> 0) & 0x00000001);

        // Get first bit in 8~15 abs quant from global memory.
        absQuant[8] |= ((tmp_char.z >> 7) & 0x00000001);
        absQuant[9] |= ((tmp_char.z >> 6) & 0x00000001);
        absQuant[10] |= ((tmp_char.z >> 5) & 0x00000001);
        absQuant[11] |= ((tmp_char.z >> 4) & 0x00000001);
        absQuant[12] |= ((tmp_char.z >> 3) & 0x00000001);
        absQuant[13] |= ((tmp_char.z >> 2) & 0x00000001);
        absQuant[14] |= ((tmp_char.z >> 1) & 0x00000001);
        absQuant[15] |= ((tmp_char.z >> 0) & 0x00000001);

        // Get first bit in 16-23 abs quant from global memory.
        absQuant[16] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[17] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[18] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[19] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[20] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[21] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[22] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[23] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[24] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[25] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[26] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[27] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[28] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[29] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[30] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[31] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[0] |= ((tmp_char.y >> 7) & 0x00000001) << (i + 1);
          absQuant[1] |= ((tmp_char.y >> 6) & 0x00000001) << (i + 1);
          absQuant[2] |= ((tmp_char.y >> 5) & 0x00000001) << (i + 1);
          absQuant[3] |= ((tmp_char.y >> 4) & 0x00000001) << (i + 1);
          absQuant[4] |= ((tmp_char.y >> 3) & 0x00000001) << (i + 1);
          absQuant[5] |= ((tmp_char.y >> 2) & 0x00000001) << (i + 1);
          absQuant[6] |= ((tmp_char.y >> 1) & 0x00000001) << (i + 1);
          absQuant[7] |= ((tmp_char.y >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[8] |= ((tmp_char.z >> 7) & 0x00000001) << (i + 1);
          absQuant[9] |= ((tmp_char.z >> 6) & 0x00000001) << (i + 1);
          absQuant[10] |= ((tmp_char.z >> 5) & 0x00000001) << (i + 1);
          absQuant[11] |= ((tmp_char.z >> 4) & 0x00000001) << (i + 1);
          absQuant[12] |= ((tmp_char.z >> 3) & 0x00000001) << (i + 1);
          absQuant[13] |= ((tmp_char.z >> 2) & 0x00000001) << (i + 1);
          absQuant[14] |= ((tmp_char.z >> 1) & 0x00000001) << (i + 1);
          absQuant[15] |= ((tmp_char.z >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[16] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[17] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[18] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[19] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[20] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[21] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[22] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[23] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get last bit in 24-31 abs quant from global memory, padding part.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001)
                        << (fixed_rate[j] - 1);
      } else if (vec_ofs == 2) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[0] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24) |
                    (0x00ff0000 & CmpDataIn[cmp_byte_ofs++] << 16);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x0000ff00 & tmp_char.x << 8) | (0x000000ff & tmp_char.y);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[0] |= ((tmp_char.z >> 7) & 0x00000001);
        absQuant[1] |= ((tmp_char.z >> 6) & 0x00000001);
        absQuant[2] |= ((tmp_char.z >> 5) & 0x00000001);
        absQuant[3] |= ((tmp_char.z >> 4) & 0x00000001);
        absQuant[4] |= ((tmp_char.z >> 3) & 0x00000001);
        absQuant[5] |= ((tmp_char.z >> 2) & 0x00000001);
        absQuant[6] |= ((tmp_char.z >> 1) & 0x00000001);
        absQuant[7] |= ((tmp_char.z >> 0) & 0x00000001);

        // Get first bit in 8~15 abs quant from global memory.
        absQuant[8] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[9] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[10] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[11] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[12] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[13] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[14] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[15] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[16] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[17] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[18] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[19] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[20] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[21] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[22] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[23] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[24] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[25] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[26] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[27] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[28] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[29] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[30] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[31] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[0] |= ((tmp_char.z >> 7) & 0x00000001) << (i + 1);
          absQuant[1] |= ((tmp_char.z >> 6) & 0x00000001) << (i + 1);
          absQuant[2] |= ((tmp_char.z >> 5) & 0x00000001) << (i + 1);
          absQuant[3] |= ((tmp_char.z >> 4) & 0x00000001) << (i + 1);
          absQuant[4] |= ((tmp_char.z >> 3) & 0x00000001) << (i + 1);
          absQuant[5] |= ((tmp_char.z >> 2) & 0x00000001) << (i + 1);
          absQuant[6] |= ((tmp_char.z >> 1) & 0x00000001) << (i + 1);
          absQuant[7] |= ((tmp_char.z >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[8] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[9] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[10] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[11] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[12] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[13] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[14] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[15] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get last bit in 16-23 abs quant from global memory, padding part.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[16] |= ((uchar_buffer >> 7) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[17] |= ((uchar_buffer >> 6) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[18] |= ((uchar_buffer >> 5) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[19] |= ((uchar_buffer >> 4) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[20] |= ((uchar_buffer >> 3) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[21] |= ((uchar_buffer >> 2) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[22] |= ((uchar_buffer >> 1) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[23] |= ((uchar_buffer >> 0) & 0x00000001)
                        << (fixed_rate[j] - 1);

        // Get last bit in 24-31 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001)
                        << (fixed_rate[j] - 1);
      } else {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[0] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x00ff0000 & tmp_char.x << 16) |
                     (0x0000ff00 & tmp_char.y << 8) | (0x000000ff & tmp_char.z);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[0] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[1] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[2] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[3] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[4] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[5] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[6] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[7] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[8] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[9] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[10] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[11] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[12] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[13] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[14] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[15] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[16] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[17] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[18] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[19] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[20] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[21] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[22] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[23] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[24] |= ((tmp_char.z >> 7) & 0x00000001) << i;
          absQuant[25] |= ((tmp_char.z >> 6) & 0x00000001) << i;
          absQuant[26] |= ((tmp_char.z >> 5) & 0x00000001) << i;
          absQuant[27] |= ((tmp_char.z >> 4) & 0x00000001) << i;
          absQuant[28] |= ((tmp_char.z >> 3) & 0x00000001) << i;
          absQuant[29] |= ((tmp_char.z >> 2) & 0x00000001) << i;
          absQuant[30] |= ((tmp_char.z >> 1) & 0x00000001) << i;
          absQuant[31] |= ((tmp_char.z >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[0] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[1] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[2] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[3] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[4] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[5] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[6] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[7] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get ith bit in 8~15 abs quant from global memory.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[8] |= ((uchar_buffer >> 7) & 0x00000001)
                       << (fixed_rate[j] - 1);
        absQuant[9] |= ((uchar_buffer >> 6) & 0x00000001)
                       << (fixed_rate[j] - 1);
        absQuant[10] |= ((uchar_buffer >> 5) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[11] |= ((uchar_buffer >> 4) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[12] |= ((uchar_buffer >> 3) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[13] |= ((uchar_buffer >> 2) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[14] |= ((uchar_buffer >> 1) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[15] |= ((uchar_buffer >> 0) & 0x00000001)
                        << (fixed_rate[j] - 1);

        // Get last bit in 16-23 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[16] |= ((uchar_buffer >> 7) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[17] |= ((uchar_buffer >> 6) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[18] |= ((uchar_buffer >> 5) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[19] |= ((uchar_buffer >> 4) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[20] |= ((uchar_buffer >> 3) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[21] |= ((uchar_buffer >> 2) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[22] |= ((uchar_buffer >> 1) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[23] |= ((uchar_buffer >> 0) & 0x00000001)
                        << (fixed_rate[j] - 1);

        // Get last bit in 24-31 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001)
                        << (fixed_rate[j] - 1);
        absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001)
                        << (fixed_rate[j] - 1);
      }
      int i = 0;
      prevQuant = 0;
      // For the .x element, reconstruct sign (absolute value), lorenzo
      // quantization, quantization, and original value.
      sign_ofs = i % 32;
      lorenQuant =
          sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
      currQuant = lorenQuant + prevQuant;
      prevQuant = currQuant;
      dec_buffer.x = currQuant * eb * 2;

      // For the .y element, reconstruct sign (absolute value), lorenzo
      // quantization, quantization, and original value.
      sign_ofs = (i + 1) % 32;
      lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i + 1] * -1
                                                      : absQuant[i + 1];
      currQuant = lorenQuant + prevQuant;
      prevQuant = currQuant;
      dec_buffer.y = currQuant * eb * 2;

      // For the .z element, reconstruct sign (absolute value), lorenzo
      // quantization, quantization, and original value.
      sign_ofs = (i + 2) % 32;
      lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i + 2] * -1
                                                      : absQuant[i + 2];
      currQuant = lorenQuant + prevQuant;
      prevQuant = currQuant;
      dec_buffer.z = currQuant * eb * 2;

      // For the .w element, reconstruct sign (absolute value), lorenzo
      // quantization, quantization, and original value.
      sign_ofs = (i + 3) % 32;
      lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i + 3] * -1
                                                      : absQuant[i + 3];
      currQuant = lorenQuant + prevQuant;
      prevQuant = currQuant;
      dec_buffer.w = currQuant * eb * 2;
      // Delorenzo and store data back to decompression data.
      reinterpret_cast<float4 *>(decData)[(base_block_start_idx + i) / 4] =
          dec_buffer;

      float4 dec_buffer_prec = dec_buffer;
#pragma unroll 8
      for (i = 4; i < 32; i += 4) {

        sign_ofs = i % 32;

        dec_buffer.x =
            sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
        dec_buffer.y = sign_flag & (1 << (31 - (sign_ofs + 1)))
                           ? absQuant[i + 1] * -1
                           : absQuant[i + 1];
        dec_buffer.z = sign_flag & (1 << (31 - (sign_ofs + 2)))
                           ? absQuant[i + 2] * -1
                           : absQuant[i + 2];
        dec_buffer.w = sign_flag & (1 << (31 - (sign_ofs + 3)))
                           ? absQuant[i + 3] * -1
                           : absQuant[i + 3];

        dec_buffer.x = (dec_buffer.x + dec_buffer_prec.x);
        dec_buffer.y = (dec_buffer.y + dec_buffer_prec.y);
        dec_buffer.z = (dec_buffer.z + dec_buffer_prec.z);
        dec_buffer.w = (dec_buffer.w + dec_buffer_prec.w);

        dec_buffer_prec = dec_buffer;

        dec_buffer.x = dec_buffer.x * eb * 2;
        dec_buffer.y = dec_buffer.y * eb * 2;
        dec_buffer.z = dec_buffer.z * eb * 2;
        dec_buffer.w = dec_buffer.w * eb * 2;

        reinterpret_cast<float4 *>(decData)[(base_block_start_idx + i) / 4] =
            dec_buffer;
        // Read data from global variable via a vectorized pattern.
      }
    }

    // Index updating across different iterations.
    cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
  }
}

// Quantization + Lorenzo Prediction
__global__ void
kernel_quant_prediction(const float *const __restrict__ localData,
                        int *const __restrict__ quantPredData, const float eb,
                        const size_t nbEle) {
  const int tid = threadIdx.x; // id of thread in block
  const int bid = blockIdx.x;
  const int idx = bid * blockDim.x + tid; // global id of thread
  const int lane = idx & 0x1f;            // id of thread in warp
  const int warp = idx >> 5;              // #n of warp in block
  const int block_num = cmp_chunk >> 5;   // #n of block which each thread
                                          // handles (1024/32 = 32)
  const float recipPrecision = 0.5f / eb;
  int base_start_idx;
  int base_block_start_idx, base_block_end_idx;
  int currQuant, prevQuant;
  float4 tmp_buffer;
  base_start_idx = warp * cmp_chunk * 32;
  for (int j = 0; j < block_num; j++) {
    // Block initilization.
    base_block_start_idx = base_start_idx + j * 1024 +
                           lane * 32; // every thread handle  32 * 32 elements
    base_block_end_idx = base_block_start_idx + 32;
    prevQuant = 0;
// Operation for each block
#pragma unroll 8
    for (int i = base_block_start_idx; i < base_block_end_idx; i += 4) {
      tmp_buffer = reinterpret_cast<const float4 *>(localData)[i / 4];

      currQuant = quantization(tmp_buffer.x, recipPrecision);
      tmp_buffer.x = currQuant - prevQuant;
      prevQuant = currQuant;
      quantPredData[i] = tmp_buffer.x;

      currQuant = quantization(tmp_buffer.y, recipPrecision);
      tmp_buffer.y = currQuant - prevQuant;
      prevQuant = currQuant;
      quantPredData[i + 1] = tmp_buffer.y;

      currQuant = quantization(tmp_buffer.z, recipPrecision);
      tmp_buffer.z = currQuant - prevQuant;
      prevQuant = currQuant;
      quantPredData[i + 2] = tmp_buffer.z;

      currQuant = quantization(tmp_buffer.w, recipPrecision);
      tmp_buffer.w = currQuant - prevQuant;
      prevQuant = currQuant;
      quantPredData[i + 3] = tmp_buffer.w;
    }
  }
}

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
                      const size_t nbEle) {
  __shared__ unsigned int excl_sum;
  __shared__ unsigned int base_idx;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid * blockDim.x + tid;
  const int lane = idx & 0x1f;
  const int warp = idx >> 5; // / 32
  const int block_num = dec_chunk >> 5;
  const int rate_ofs = (nbEle + dec_tblock_size * dec_chunk - 1) /
                       (dec_tblock_size * dec_chunk) *
                       (dec_tblock_size * dec_chunk) / 32;
  int base_start_idx;
  int base_block_start_idx;
  unsigned int sign_flag_cmp[block_num];
  int block_idx;
  int maxQuant = 0;
  unsigned int thread_ofs2 = 0;
  int maxQuan2 = 0;
  int outlier;
  int absQuant[dec_chunk];
  int fixed_rate_cmp[block_num];
  int lorenQuant;
  int fixed_rate[block_num];
  unsigned int thread_ofs = 0;
  uchar4 tmp_char;
  // Obtain fixed rate information for each block.
  for (int j = 0; j < block_num; j++) {
    block_idx = warp * dec_chunk + j * 32 + lane;
    fixed_rate[j] = (int)CmpDataIn[block_idx];
    // Encoding selection.
    int encoding_selection = fixed_rate[j] >> 7; // use outlier encoding or not
    int outlier = ((fixed_rate[j] & 0x60) >> 5) +
                  1; // 5'6' bit to encode byte for the outlier
    int temp_rate = fixed_rate[j] &
                    0x1f; // first 5 bit to encode length of rest of the block
    if (!encoding_selection)
      thread_ofs += temp_rate ? (4 + temp_rate * 4) : 0;
    else
      thread_ofs += 4 + temp_rate * 4 + outlier;
    __syncthreads();
  }

  // Warp-level prefix-sum (inclusive), also thread-block-level.
#pragma unroll 5
  for (int i = 1; i < 32; i <<= 1) {
    int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
    if (lane >= i)
      thread_ofs += tmp;
  }
  __syncthreads();

  // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
  if (lane == 31) {
    locOffsetIn[warp + 1] = thread_ofs;
    __threadfence();
    if (warp == 0) {
      flag[0] = 2;
      __threadfence();
      flag[1] = 1;
      __threadfence();
    } else {
      flag[warp + 1] = 1;
      __threadfence();
    }
  }
  __syncthreads();

  // Global-level prefix-sum (exclusive).
  if (warp > 0) {
    if (!lane) {
      // Decoupled look-back
      int lookback = warp;
      int loc_excl_sum = 0;
      while (lookback > 0) {
        int status;
        // Local sum not end.
        do {
          status = flag[lookback];
          __threadfence();
        } while (status == 0);
        // Lookback end.
        if (status == 2) {
          loc_excl_sum += CmpOffsetIn[lookback];
          __threadfence();
          break;
        }
        // Continues lookback.
        if (status == 1)
          loc_excl_sum += locOffsetIn[lookback];
        lookback--;
        __threadfence();
      }
      excl_sum = loc_excl_sum;
    }
    __syncthreads();
  }

  if (warp > 0) {
    // Update global flag.
    if (!lane)
      CmpOffsetIn[warp] = excl_sum;
    __threadfence();
    if (!lane)
      flag[warp] = 2;
    __threadfence();
  }
  __syncthreads();

  // Retrieving compression bytes and reconstruct decompression data.
  if (!lane)
    base_idx = excl_sum + rate_ofs;
  __syncthreads();
  // Restore bit-shuffle for each block.
  unsigned int base_cmp_byte_ofs = base_idx;
  unsigned int cmp_byte_ofs;
  unsigned int tmp_byte_ofs = 0;
  unsigned int cur_byte_ofs = 0;
  base_start_idx = warp * dec_chunk * 32;
  for (int j = 0; j < block_num; j++) {
    // Initialization, guiding encoding process.
    int encoding_selection = fixed_rate[j] >> 7;
    int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
    fixed_rate[j] &= 0x1f;
    int outlier_buffer = 0;
    fixed_rate_cmp[j] = 0;
    sign_flag_cmp[j] = 0;
    base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
    unsigned int sign_flag = 0;

    // Restore index for j-th iteration.
    if (!encoding_selection)
      tmp_byte_ofs = (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
    else
      tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
      int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
      if (lane >= i)
        tmp_byte_ofs += tmp;
    }
    unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
    if (!lane)
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
    else
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

    // If outlier encoding, retrieve outliers here.
    if (encoding_selection) {
      for (int i = 0; i < outlier_byte_num; i++) {
        int buffer = CmpDataIn[cmp_byte_ofs++] << (8 * i);
        outlier_buffer |= buffer;
      }

      // Corner case: all data points except outliers are 0.
      if (!fixed_rate[j]) {
        sign_flag = (0xff000000 & (CmpDataIn[cmp_byte_ofs++] << 24)) |
                    (0x00ff0000 & (CmpDataIn[cmp_byte_ofs++] << 16)) |
                    (0x0000ff00 & (CmpDataIn[cmp_byte_ofs++] << 8)) |
                    (0x000000ff & CmpDataIn[cmp_byte_ofs++]);
        maxQuant = 0;
        maxQuan2 = 0;
        lorenQuant = outlier_buffer + predQuant[base_block_start_idx];
        sign_flag_cmp[j] |= (lorenQuant < 0) << (31);
        absQuant[j * 32] = abs(lorenQuant);
        maxQuant = max(maxQuant, lorenQuant);
        outlier = absQuant[j * 32];
        for (int i = 1; i < 32; i++) {
          sign_flag_cmp[j] |= (predQuant[base_block_start_idx + i] < 0)
                              << (31 - (i));
          lorenQuant = predQuant[base_block_start_idx + i];
          absQuant[j * 32 + i] = abs(lorenQuant);
          maxQuant = max(maxQuant, absQuant[j * 32 + i]);
          maxQuan2 = max(maxQuan2, absQuant[j * 32 + i]);
        }
      }
    }

    // Operation for each block, if zero block then do nothing.
    if (fixed_rate[j]) {
      // Padding vector operation for reverse outlier encoding.
      int vec_ofs = cmp_byte_ofs % 4;
      if (vec_ofs == 0) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                    (0x00ff0000 & (tmp_char.y << 16)) |
                    (0x0000ff00 & (tmp_char.z << 8)) |
                    (0x000000ff & tmp_char.w);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j]; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
        }
      } else if (vec_ofs == 1) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24) |
                    (0x00ff0000 & CmpDataIn[cmp_byte_ofs++] << 16) |
                    (0x0000ff00 & CmpDataIn[cmp_byte_ofs++] << 8);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x000000ff & tmp_char.x);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[j * 32] |= ((tmp_char.y >> 7) & 0x00000001);
        absQuant[j * 32 + 1] |= ((tmp_char.y >> 6) & 0x00000001);
        absQuant[j * 32 + 2] |= ((tmp_char.y >> 5) & 0x00000001);
        absQuant[j * 32 + 3] |= ((tmp_char.y >> 4) & 0x00000001);
        absQuant[j * 32 + 4] |= ((tmp_char.y >> 3) & 0x00000001);
        absQuant[j * 32 + 5] |= ((tmp_char.y >> 2) & 0x00000001);
        absQuant[j * 32 + 6] |= ((tmp_char.y >> 1) & 0x00000001);
        absQuant[j * 32 + 7] |= ((tmp_char.y >> 0) & 0x00000001);

        // Get first bit in 8~15 abs quant from global memory.
        absQuant[j * 32 + 8] |= ((tmp_char.z >> 7) & 0x00000001);
        absQuant[j * 32 + 9] |= ((tmp_char.z >> 6) & 0x00000001);
        absQuant[j * 32 + 10] |= ((tmp_char.z >> 5) & 0x00000001);
        absQuant[j * 32 + 11] |= ((tmp_char.z >> 4) & 0x00000001);
        absQuant[j * 32 + 12] |= ((tmp_char.z >> 3) & 0x00000001);
        absQuant[j * 32 + 13] |= ((tmp_char.z >> 2) & 0x00000001);
        absQuant[j * 32 + 14] |= ((tmp_char.z >> 1) & 0x00000001);
        absQuant[j * 32 + 15] |= ((tmp_char.z >> 0) & 0x00000001);

        // Get first bit in 16-23 abs quant from global memory.
        absQuant[j * 32 + 16] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[j * 32 + 17] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[j * 32 + 18] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[j * 32 + 19] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[j * 32 + 20] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[j * 32 + 21] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[j * 32 + 22] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[j * 32 + 23] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.y >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 1] |= ((tmp_char.y >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 2] |= ((tmp_char.y >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 3] |= ((tmp_char.y >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 4] |= ((tmp_char.y >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 5] |= ((tmp_char.y >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 6] |= ((tmp_char.y >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 7] |= ((tmp_char.y >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.z >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 9] |= ((tmp_char.z >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 10] |= ((tmp_char.z >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 11] |= ((tmp_char.z >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 12] |= ((tmp_char.z >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 13] |= ((tmp_char.z >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 14] |= ((tmp_char.z >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 15] |= ((tmp_char.z >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 17] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 18] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 19] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 20] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 21] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 22] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 23] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get last bit in 24-31 abs quant from global memory, padding part.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 24] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 25] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 26] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 27] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 28] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 29] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 30] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 31] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);
      } else if (vec_ofs == 2) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24) |
                    (0x00ff0000 & CmpDataIn[cmp_byte_ofs++] << 16);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x0000ff00 & tmp_char.x << 8) | (0x000000ff & tmp_char.y);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[j * 32] |= ((tmp_char.z >> 7) & 0x00000001);
        absQuant[j * 32 + 1] |= ((tmp_char.z >> 6) & 0x00000001);
        absQuant[j * 32 + 2] |= ((tmp_char.z >> 5) & 0x00000001);
        absQuant[j * 32 + 3] |= ((tmp_char.z >> 4) & 0x00000001);
        absQuant[j * 32 + 4] |= ((tmp_char.z >> 3) & 0x00000001);
        absQuant[j * 32 + 5] |= ((tmp_char.z >> 2) & 0x00000001);
        absQuant[j * 32 + 6] |= ((tmp_char.z >> 1) & 0x00000001);
        absQuant[j * 32 + 7] |= ((tmp_char.z >> 0) & 0x00000001);

        // Get first bit in 8~15 abs quant from global memory.
        absQuant[j * 32 + 8] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[j * 32 + 9] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[j * 32 + 10] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[j * 32 + 11] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[j * 32 + 12] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[j * 32 + 13] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[j * 32 + 14] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[j * 32 + 15] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 17] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 18] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 19] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 20] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 21] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 22] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 23] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.z >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 1] |= ((tmp_char.z >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 2] |= ((tmp_char.z >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 3] |= ((tmp_char.z >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 4] |= ((tmp_char.z >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 5] |= ((tmp_char.z >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 6] |= ((tmp_char.z >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 7] |= ((tmp_char.z >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 9] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 10] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 11] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 12] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 13] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 14] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 15] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get last bit in 16-23 abs quant from global memory, padding part.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 16] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 17] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 18] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 19] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 20] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 21] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 22] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 23] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);

        // Get last bit in 24-31 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 24] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 25] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 26] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 27] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 28] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 29] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 30] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 31] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);
      } else {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x00ff0000 & tmp_char.x << 16) |
                     (0x0000ff00 & tmp_char.y << 8) | (0x000000ff & tmp_char.z);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[j * 32] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[j * 32 + 1] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[j * 32 + 2] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[j * 32 + 3] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[j * 32 + 4] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[j * 32 + 5] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[j * 32 + 6] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[j * 32 + 7] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 9] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 10] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 11] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 12] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 13] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 14] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 15] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 17] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 18] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 19] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 20] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 21] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 22] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 23] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.z >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.z >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.z >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.z >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.z >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.z >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.z >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.z >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 1] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 2] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 3] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 4] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 5] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 6] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 7] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get ith bit in 8~15 abs quant from global memory.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 8] |= ((uchar_buffer >> 7) & 0x00000001)
                                << (fixed_rate[j] - 1);
        absQuant[j * 32 + 9] |= ((uchar_buffer >> 6) & 0x00000001)
                                << (fixed_rate[j] - 1);
        absQuant[j * 32 + 10] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 11] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 12] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 13] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 14] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 15] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);

        // Get last bit in 16-23 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 16] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 17] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 18] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 19] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 20] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 21] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 22] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 23] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);

        // Get last bit in 24-31 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 24] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 25] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 26] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 27] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 28] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 29] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 30] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 31] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);
      }
      //  Decompress and sum with predQuant.
      maxQuan2 = 0;
      maxQuant = 0;
      lorenQuant = ((sign_flag & (1 << (31))) ? absQuant[j * 32] * -1
                                              : absQuant[j * 32]) +
                   predQuant[base_block_start_idx];
      sign_flag_cmp[j] |= (lorenQuant < 0) << 31;
      absQuant[j * 32] = abs(lorenQuant);
      maxQuant = max(maxQuant, absQuant[j * 32]);
      outlier = absQuant[j * 32];
      // TODO: unroll
      for (int i = 1; i < 32; i++) {
        lorenQuant = ((sign_flag & (1 << (31 - i))) ? absQuant[j * 32 + i] * -1
                                                    : absQuant[j * 32 + i]) +
                     predQuant[base_block_start_idx + i];
        sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - i);
        absQuant[j * 32 + i] = abs(lorenQuant);
        maxQuant = max(maxQuant, absQuant[j * 32 + i]);
        maxQuan2 = max(maxQuan2, absQuant[j * 32 + i]);
      }
    }

    int fr1 = get_bit_num(maxQuant);
    int fr2 = get_bit_num(maxQuan2);
    outlier = (get_bit_num(outlier) + 7) / 8;
    int temp_rate = 0;
    int temp_ofs1 = fr1 ? 4 + fr1 * 4 : 0;
    int temp_ofs2 = fr2 ? 4 + fr2 * 4 + outlier : 4 + outlier;
    if (temp_ofs1 <= temp_ofs2) {
      thread_ofs2 += temp_ofs1;
      temp_rate = fr1;
    } else {
      thread_ofs2 += temp_ofs2;
      temp_rate = fr2 | 0x80 | ((outlier - 1) << 5);
    }
    fixed_rate_cmp[j] = temp_rate;
    CmpDataOut[(base_block_start_idx / 32)] =
        (unsigned char)fixed_rate_cmp[j]; // ERROR
    __syncthreads();
    // Index updating across different iterations.
    cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
  }

#pragma unroll 5
  for (int i = 1; i < 32; i <<= 1) {
    int tmp = __shfl_up_sync(0xffffffff, thread_ofs2, i);
    if (lane >= i)
      thread_ofs2 += tmp;
  }
  __syncthreads();

  // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
  if (lane == 31) {
    locOffsetOut[warp + 1] = thread_ofs2;
    __threadfence();
    if (warp == 0) {
      flag_cmp[0] = 2;
      __threadfence();
      flag_cmp[1] = 1;
      __threadfence();
    } else {
      flag_cmp[warp + 1] = 1;
      __threadfence();
    }
  }
  __syncthreads();

  // Global-level prefix-sum (exclusive).
  if (warp > 0) {
    if (!lane) {
      // Decoupled look-back
      int lookback = warp;
      int loc_excl_sum = 0;
      while (lookback > 0) {
        int status;
        // Local sum not end.
        do {
          status = flag_cmp[lookback];
          __threadfence();
        } while (status == 0);
        // Lookback end.
        if (status == 2) {
          loc_excl_sum += CmpOffsetOut[lookback];
          __threadfence();
          break;
        }
        // Continues lookback.
        if (status == 1)
          loc_excl_sum += locOffsetOut[lookback];
        lookback--;
        __threadfence();
      }
      excl_sum = loc_excl_sum;
    }
    __syncthreads();
  }

  if (warp > 0) {
    // Update global flag.
    if (!lane)
      CmpOffsetOut[warp] = excl_sum;
    __threadfence();
    if (!lane)
      flag_cmp[warp] = 2;
    __threadfence();
  }
  __syncthreads();

  // Assigning compression bytes by given prefix-sum results.
  if (!lane)
    base_idx = excl_sum + rate_ofs;
  __syncthreads();
  base_cmp_byte_ofs = base_idx;
  tmp_byte_ofs = 0;
  cur_byte_ofs = 0;
  for (int j = 0; j < block_num; j++) {
    // Initialization, guiding encoding process.
    int encoding_selection = fixed_rate_cmp[j] >> 7;
    int outlier_byte_num = ((fixed_rate_cmp[j] & 0x60) >> 5) + 1;
    fixed_rate_cmp[j] &= 0x1f;
    int chunk_idx_start = j * 32;

    // Restore index for j-th iteration.
    if (!encoding_selection)
      tmp_byte_ofs = (fixed_rate_cmp[j]) ? (4 + fixed_rate_cmp[j] * 4) : 0;
    else
      tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate_cmp[j] * 4;
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
      int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
      if (lane >= i)
        tmp_byte_ofs += tmp;
    }
    unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
    if (!lane)
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
    else
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

    // If outlier encoding, storing outliers here.
    if (encoding_selection) {
      for (int i = 0; i < outlier_byte_num; i++) {
        CmpDataOut[cmp_byte_ofs++] =
            (unsigned char)(absQuant[chunk_idx_start] & 0xff);
        absQuant[chunk_idx_start] >>= 8;
      }

      // Corner case: all data points except outliers are 0.
      if (!fixed_rate_cmp[j]) {
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 16);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 8);
        CmpDataOut[cmp_byte_ofs++] = 0xff & sign_flag_cmp[j];
      }
    }

    // Operation for each block, if zero block then do nothing.
    if (fixed_rate_cmp[j]) {
      // Padding vector operation for outlier encoding.
      int vec_ofs = cmp_byte_ofs % 4;
      if (vec_ofs == 0) {
        // Assign sign information for one block.
        tmp_char.x = 0xff & (sign_flag_cmp[j] >> 24);
        tmp_char.y = 0xff & (sign_flag_cmp[j] >> 16);
        tmp_char.z = 0xff & (sign_flag_cmp[j] >> 8);
        tmp_char.w = 0xff & sign_flag_cmp[j];
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j]; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.x = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.x = tmp_char.x |
                       (((absQuant[chunk_idx_start + 0] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 1] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 2] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 3] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 4] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 5] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 6] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 7] & mask) >> i) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.y = (((absQuant[chunk_idx_start + 8] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 9] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 10] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 11] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 12] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 13] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 14] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 15] & mask) >> i) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.z = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.w = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
          mask <<= 1;
        }
      } else if (vec_ofs == 1) {
        // Assign sign information for one block, padding part.
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 16);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 8);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & sign_flag_cmp[j];
        if (!encoding_selection)
          tmp_char.y = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.y = tmp_char.y | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        tmp_char.z = ((absQuant[chunk_idx_start + 8] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 9] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 10] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 11] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 12] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 13] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 14] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 15] & 1) << 0);
        tmp_char.w = ((absQuant[chunk_idx_start + 16] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 17] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 18] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 19] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 20] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 21] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 22] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 23] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.x = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.y = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.y =
              tmp_char.y |
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.z =
              (((absQuant[chunk_idx_start + 8] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 9] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 10] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 11] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 12] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 13] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 14] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 15] & mask) >> (i + 1)) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.w =
              (((absQuant[chunk_idx_start + 16] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 17] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 18] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 19] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 20] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 21] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 22] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 23] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
      } else if (vec_ofs == 2) {
        // Assign sign information for one block, padding part.
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 16);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & (sign_flag_cmp[j] >> 8);
        tmp_char.y = 0xff & sign_flag_cmp[j];
        if (!encoding_selection)
          tmp_char.z = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.z = tmp_char.z | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        tmp_char.w = ((absQuant[chunk_idx_start + 8] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 9] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 10] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 11] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 12] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 13] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 14] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 15] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.x = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.y = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.z = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.z =
              tmp_char.z |
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.w =
              (((absQuant[chunk_idx_start + 8] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 9] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 10] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 11] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 12] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 13] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 14] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 15] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 16] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 17] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 18] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 19] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 20] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 21] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 22] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 23] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
      } else {
        // Assign sign information for one block, padding part.
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & (sign_flag_cmp[j] >> 16);
        tmp_char.y = 0xff & (sign_flag_cmp[j] >> 8);
        tmp_char.z = 0xff & sign_flag_cmp[j];
        if (!encoding_selection)
          tmp_char.w = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.w = tmp_char.w | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.x = (((absQuant[chunk_idx_start + 8] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 9] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 10] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 11] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 12] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 13] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 14] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 15] & mask) >> i) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.y = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.z = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.w = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.w =
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 8] & mask) >> (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 9] & mask) >> (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 10] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 11] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 12] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 13] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 14] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 15] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 16] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 17] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 18] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 19] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 20] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 21] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 22] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 23] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
      }
    }

    cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
  }
}

__global__ void
kernel_homomophic_sum_F(const unsigned char *const __restrict__ CmpDataIn,
                        volatile unsigned int *const __restrict__ CmpOffsetIn,
                        unsigned char *const __restrict__ CmpDataOut,
                        volatile unsigned int *const __restrict__ locOffsetOut,
                        volatile unsigned int *const __restrict__ CmpOffsetOut,
                        volatile unsigned int *const __restrict__ locOffsetIn,
                        volatile int *const __restrict__ flag,
                        volatile int *const __restrict__ flag_cmp,
                        const float *const __restrict__ localChunk,
                        const float eb, const size_t nbEle) {
  __shared__ unsigned int excl_sum;
  __shared__ unsigned int base_idx;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid * blockDim.x + tid;
  const int lane = idx & 0x1f;
  const int warp = idx >> 5; // / 32
  const int block_num = dec_chunk >> 5;
  const int rate_ofs = (nbEle + dec_tblock_size * dec_chunk - 1) /
                       (dec_tblock_size * dec_chunk) *
                       (dec_tblock_size * dec_chunk) / 32;
  int base_start_idx;
  int base_block_start_idx;
  const float recipPrecision = 0.5f / eb;
  unsigned int sign_flag_cmp[block_num];
  int block_idx;
  int maxQuant = 0;
  unsigned int thread_ofs2 = 0;
  int maxQuan2 = 0;
  int outlier;
  int absQuant[dec_chunk];
  int fixed_rate_cmp[block_num];
  int lorenQuant;
  int fixed_rate[block_num];
  unsigned int thread_ofs = 0;
  uchar4 tmp_char;
  // Obtain fixed rate information for each block.
  for (int j = 0; j < block_num; j++) {
    block_idx = warp * dec_chunk + j * 32 + lane;
    fixed_rate[j] = (int)CmpDataIn[block_idx];
    // Encoding selection.
    int encoding_selection = fixed_rate[j] >> 7; // use outlier encoding or not
    int outlier = ((fixed_rate[j] & 0x60) >> 5) +
                  1; // 5'6' bit to encode byte for the outlier
    int temp_rate = fixed_rate[j] &
                    0x1f; // first 5 bit to encode length of rest of the block
    if (!encoding_selection)
      thread_ofs += temp_rate ? (4 + temp_rate * 4) : 0;
    else
      thread_ofs += 4 + temp_rate * 4 + outlier;
    __syncthreads();
  }

  // Warp-level prefix-sum (inclusive), also thread-block-level.
#pragma unroll 5
  for (int i = 1; i < 32; i <<= 1) {
    int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
    if (lane >= i)
      thread_ofs += tmp;
  }
  __syncthreads();

  // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
  if (lane == 31) {
    locOffsetIn[warp + 1] = thread_ofs;
    __threadfence();
    if (warp == 0) {
      flag[0] = 2;
      __threadfence();
      flag[1] = 1;
      __threadfence();
    } else {
      flag[warp + 1] = 1;
      __threadfence();
    }
  }
  __syncthreads();

  // Global-level prefix-sum (exclusive).
  if (warp > 0) {
    if (!lane) {
      // Decoupled look-back
      int lookback = warp;
      int loc_excl_sum = 0;
      while (lookback > 0) {
        int status;
        // Local sum not end.
        do {
          status = flag[lookback];
          __threadfence();
        } while (status == 0);
        // Lookback end.
        if (status == 2) {
          loc_excl_sum += CmpOffsetIn[lookback];
          __threadfence();
          break;
        }
        // Continues lookback.
        if (status == 1)
          loc_excl_sum += locOffsetIn[lookback];
        lookback--;
        __threadfence();
      }
      excl_sum = loc_excl_sum;
    }
    __syncthreads();
  }

  if (warp > 0) {
    // Update global flag.
    if (!lane)
      CmpOffsetIn[warp] = excl_sum;
    __threadfence();
    if (!lane)
      flag[warp] = 2;
    __threadfence();
  }
  __syncthreads();

  // Retrieving compression bytes and reconstruct decompression data.
  if (!lane)
    base_idx = excl_sum + rate_ofs;
  __syncthreads();
  // Restore bit-shuffle for each block.
  unsigned int base_cmp_byte_ofs = base_idx;
  unsigned int cmp_byte_ofs;
  unsigned int tmp_byte_ofs = 0;
  unsigned int cur_byte_ofs = 0;
  base_start_idx = warp * dec_chunk * 32;
  for (int j = 0; j < block_num; j++) {
    // Initialization, guiding encoding process.
    int encoding_selection = fixed_rate[j] >> 7;
    int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
    fixed_rate[j] &= 0x1f;
    int outlier_buffer = 0;
    fixed_rate_cmp[j] = 0;
    sign_flag_cmp[j] = 0;
    base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
    unsigned int sign_flag = 0;

    // Restore index for j-th iteration.
    if (!encoding_selection)
      tmp_byte_ofs = (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
    else
      tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
      int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
      if (lane >= i)
        tmp_byte_ofs += tmp;
    }
    unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
    if (!lane)
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
    else
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

    // If outlier encoding, retrieve outliers here.
    if (encoding_selection) {
      for (int i = 0; i < outlier_byte_num; i++) {
        int buffer = CmpDataIn[cmp_byte_ofs++] << (8 * i);
        outlier_buffer |= buffer;
      }

      // Corner case: all data points except outliers are 0.
      if (!fixed_rate[j]) {
        sign_flag = (0xff000000 & (CmpDataIn[cmp_byte_ofs++] << 24)) |
                    (0x00ff0000 & (CmpDataIn[cmp_byte_ofs++] << 16)) |
                    (0x0000ff00 & (CmpDataIn[cmp_byte_ofs++] << 8)) |
                    (0x000000ff & CmpDataIn[cmp_byte_ofs++]);
        maxQuant = 0;
        maxQuan2 = 0;
        int current_quant = 0;
        int prevQuant = 0;
        float4 tmp_buffer;
#pragma unroll 8
        for (int i = 0; i < 32; i += 4) {
          tmp_buffer = reinterpret_cast<const float4 *>(
              localChunk)[(base_block_start_idx + i) / 4];
          current_quant = quantization(tmp_buffer.x, recipPrecision);
          lorenQuant = current_quant - prevQuant;
          prevQuant = current_quant;
          if (i) {
            absQuant[j * 32 + i] = abs(lorenQuant);
            maxQuan2 = max(maxQuan2, absQuant[j * 32 + i]);
          } else {
            lorenQuant += outlier;
            absQuant[j * 32 + i] = abs(lorenQuant);
          }
          maxQuant = max(maxQuant, absQuant[j * 32 + i]);
          sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - i);

          current_quant = quantization(tmp_buffer.y, recipPrecision);
          lorenQuant = current_quant - prevQuant;
          prevQuant = current_quant;
          absQuant[j * 32 + i + 1] = abs(lorenQuant);
          maxQuan2 = max(maxQuan2, absQuant[j * 32 + i + 1]);
          maxQuant = max(maxQuant, absQuant[j * 32 + i + 1]);
          sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - (i + 1));

          current_quant = quantization(tmp_buffer.z, recipPrecision);
          lorenQuant = current_quant - prevQuant;
          prevQuant = current_quant;
          absQuant[j * 32 + i + 2] = abs(lorenQuant);
          maxQuan2 = max(maxQuan2, absQuant[j * 32 + i + 2]);
          maxQuant = max(maxQuant, absQuant[j * 32 + i + 2]);
          sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - (i + 2));

          current_quant = quantization(tmp_buffer.w, recipPrecision);
          lorenQuant = current_quant - prevQuant;
          prevQuant = current_quant;
          absQuant[j * 32 + i + 3] = abs(lorenQuant);
          maxQuan2 = max(maxQuan2, absQuant[j * 32 + i + 3]);
          maxQuant = max(maxQuant, absQuant[j * 32 + i + 3]);
          sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - (i + 3));
        }
        outlier = absQuant[j * 32];
      }
    }

    // Operation for each block, if zero block then do nothing.
    if (fixed_rate[j]) {
      // Padding vector operation for reverse outlier encoding.
      int vec_ofs = cmp_byte_ofs % 4;
      if (vec_ofs == 0) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                    (0x00ff0000 & (tmp_char.y << 16)) |
                    (0x0000ff00 & (tmp_char.z << 8)) |
                    (0x000000ff & tmp_char.w);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j]; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
        }
      } else if (vec_ofs == 1) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24) |
                    (0x00ff0000 & CmpDataIn[cmp_byte_ofs++] << 16) |
                    (0x0000ff00 & CmpDataIn[cmp_byte_ofs++] << 8);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x000000ff & tmp_char.x);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[j * 32] |= ((tmp_char.y >> 7) & 0x00000001);
        absQuant[j * 32 + 1] |= ((tmp_char.y >> 6) & 0x00000001);
        absQuant[j * 32 + 2] |= ((tmp_char.y >> 5) & 0x00000001);
        absQuant[j * 32 + 3] |= ((tmp_char.y >> 4) & 0x00000001);
        absQuant[j * 32 + 4] |= ((tmp_char.y >> 3) & 0x00000001);
        absQuant[j * 32 + 5] |= ((tmp_char.y >> 2) & 0x00000001);
        absQuant[j * 32 + 6] |= ((tmp_char.y >> 1) & 0x00000001);
        absQuant[j * 32 + 7] |= ((tmp_char.y >> 0) & 0x00000001);

        // Get first bit in 8~15 abs quant from global memory.
        absQuant[j * 32 + 8] |= ((tmp_char.z >> 7) & 0x00000001);
        absQuant[j * 32 + 9] |= ((tmp_char.z >> 6) & 0x00000001);
        absQuant[j * 32 + 10] |= ((tmp_char.z >> 5) & 0x00000001);
        absQuant[j * 32 + 11] |= ((tmp_char.z >> 4) & 0x00000001);
        absQuant[j * 32 + 12] |= ((tmp_char.z >> 3) & 0x00000001);
        absQuant[j * 32 + 13] |= ((tmp_char.z >> 2) & 0x00000001);
        absQuant[j * 32 + 14] |= ((tmp_char.z >> 1) & 0x00000001);
        absQuant[j * 32 + 15] |= ((tmp_char.z >> 0) & 0x00000001);

        // Get first bit in 16-23 abs quant from global memory.
        absQuant[j * 32 + 16] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[j * 32 + 17] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[j * 32 + 18] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[j * 32 + 19] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[j * 32 + 20] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[j * 32 + 21] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[j * 32 + 22] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[j * 32 + 23] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.y >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 1] |= ((tmp_char.y >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 2] |= ((tmp_char.y >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 3] |= ((tmp_char.y >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 4] |= ((tmp_char.y >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 5] |= ((tmp_char.y >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 6] |= ((tmp_char.y >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 7] |= ((tmp_char.y >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.z >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 9] |= ((tmp_char.z >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 10] |= ((tmp_char.z >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 11] |= ((tmp_char.z >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 12] |= ((tmp_char.z >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 13] |= ((tmp_char.z >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 14] |= ((tmp_char.z >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 15] |= ((tmp_char.z >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 17] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 18] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 19] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 20] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 21] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 22] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 23] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get last bit in 24-31 abs quant from global memory, padding part.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 24] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 25] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 26] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 27] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 28] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 29] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 30] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 31] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);
      } else if (vec_ofs == 2) {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24) |
                    (0x00ff0000 & CmpDataIn[cmp_byte_ofs++] << 16);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x0000ff00 & tmp_char.x << 8) | (0x000000ff & tmp_char.y);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[j * 32] |= ((tmp_char.z >> 7) & 0x00000001);
        absQuant[j * 32 + 1] |= ((tmp_char.z >> 6) & 0x00000001);
        absQuant[j * 32 + 2] |= ((tmp_char.z >> 5) & 0x00000001);
        absQuant[j * 32 + 3] |= ((tmp_char.z >> 4) & 0x00000001);
        absQuant[j * 32 + 4] |= ((tmp_char.z >> 3) & 0x00000001);
        absQuant[j * 32 + 5] |= ((tmp_char.z >> 2) & 0x00000001);
        absQuant[j * 32 + 6] |= ((tmp_char.z >> 1) & 0x00000001);
        absQuant[j * 32 + 7] |= ((tmp_char.z >> 0) & 0x00000001);

        // Get first bit in 8~15 abs quant from global memory.
        absQuant[j * 32 + 8] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[j * 32 + 9] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[j * 32 + 10] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[j * 32 + 11] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[j * 32 + 12] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[j * 32 + 13] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[j * 32 + 14] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[j * 32 + 15] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 17] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 18] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 19] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 20] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 21] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 22] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 23] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.z >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 1] |= ((tmp_char.z >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 2] |= ((tmp_char.z >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 3] |= ((tmp_char.z >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 4] |= ((tmp_char.z >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 5] |= ((tmp_char.z >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 6] |= ((tmp_char.z >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 7] |= ((tmp_char.z >> 0) & 0x00000001) << (i + 1);

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 9] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 10] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 11] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 12] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 13] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 14] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 15] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get last bit in 16-23 abs quant from global memory, padding part.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 16] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 17] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 18] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 19] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 20] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 21] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 22] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 23] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);

        // Get last bit in 24-31 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 24] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 25] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 26] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 27] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 28] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 29] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 30] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 31] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);
      } else {
        // Retrieve quant data for one block.
        for (int i = 0; i < 32; i++)
          absQuant[j * 32 + i] = 0;

        // Operation for outlier encoding method.
        if (encoding_selection)
          absQuant[j * 32] = outlier_buffer;

        // Retrieve sign information for one block, padding part.
        sign_flag = (0xff000000 & CmpDataIn[cmp_byte_ofs++] << 24);

        // Retrieve sign information for one block and reverse block
        // bit-shuffle, padding for vectorization.
        tmp_char =
            reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
        sign_flag |= (0x00ff0000 & tmp_char.x << 16) |
                     (0x0000ff00 & tmp_char.y << 8) | (0x000000ff & tmp_char.z);

        // Get first bit in 0~7 abs quant from global memory.
        if (!encoding_selection)
          absQuant[j * 32] |= ((tmp_char.w >> 7) & 0x00000001);
        absQuant[j * 32 + 1] |= ((tmp_char.w >> 6) & 0x00000001);
        absQuant[j * 32 + 2] |= ((tmp_char.w >> 5) & 0x00000001);
        absQuant[j * 32 + 3] |= ((tmp_char.w >> 4) & 0x00000001);
        absQuant[j * 32 + 4] |= ((tmp_char.w >> 3) & 0x00000001);
        absQuant[j * 32 + 5] |= ((tmp_char.w >> 2) & 0x00000001);
        absQuant[j * 32 + 6] |= ((tmp_char.w >> 1) & 0x00000001);
        absQuant[j * 32 + 7] |= ((tmp_char.w >> 0) & 0x00000001);
        cmp_byte_ofs += 4;

        // Reverse block bit-shuffle.
        for (int i = 0; i < fixed_rate[j] - 1; i++) {
          // Initialization.
          tmp_char =
              reinterpret_cast<const uchar4 *>(CmpDataIn)[cmp_byte_ofs / 4];
          cmp_byte_ofs += 4;

          // Get ith bit in 8~15 abs quant from global memory.
          absQuant[j * 32 + 8] |= ((tmp_char.x >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 9] |= ((tmp_char.x >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 10] |= ((tmp_char.x >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 11] |= ((tmp_char.x >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 12] |= ((tmp_char.x >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 13] |= ((tmp_char.x >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 14] |= ((tmp_char.x >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 15] |= ((tmp_char.x >> 0) & 0x00000001) << i;

          // Get ith bit in 16-23 abs quant from global memory.
          absQuant[j * 32 + 16] |= ((tmp_char.y >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 17] |= ((tmp_char.y >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 18] |= ((tmp_char.y >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 19] |= ((tmp_char.y >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 20] |= ((tmp_char.y >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 21] |= ((tmp_char.y >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 22] |= ((tmp_char.y >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 23] |= ((tmp_char.y >> 0) & 0x00000001) << i;

          // Get ith bit in 24-31 abs quant from global memory.
          absQuant[j * 32 + 24] |= ((tmp_char.z >> 7) & 0x00000001) << i;
          absQuant[j * 32 + 25] |= ((tmp_char.z >> 6) & 0x00000001) << i;
          absQuant[j * 32 + 26] |= ((tmp_char.z >> 5) & 0x00000001) << i;
          absQuant[j * 32 + 27] |= ((tmp_char.z >> 4) & 0x00000001) << i;
          absQuant[j * 32 + 28] |= ((tmp_char.z >> 3) & 0x00000001) << i;
          absQuant[j * 32 + 29] |= ((tmp_char.z >> 2) & 0x00000001) << i;
          absQuant[j * 32 + 30] |= ((tmp_char.z >> 1) & 0x00000001) << i;
          absQuant[j * 32 + 31] |= ((tmp_char.z >> 0) & 0x00000001) << i;

          // Get ith bit in 0~7 abs quant from global memory.
          if (!encoding_selection)
            absQuant[j * 32] |= ((tmp_char.w >> 7) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 1] |= ((tmp_char.w >> 6) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 2] |= ((tmp_char.w >> 5) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 3] |= ((tmp_char.w >> 4) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 4] |= ((tmp_char.w >> 3) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 5] |= ((tmp_char.w >> 2) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 6] |= ((tmp_char.w >> 1) & 0x00000001) << (i + 1);
          absQuant[j * 32 + 7] |= ((tmp_char.w >> 0) & 0x00000001) << (i + 1);
        }

        // Get ith bit in 8~15 abs quant from global memory.
        int uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 8] |= ((uchar_buffer >> 7) & 0x00000001)
                                << (fixed_rate[j] - 1);
        absQuant[j * 32 + 9] |= ((uchar_buffer >> 6) & 0x00000001)
                                << (fixed_rate[j] - 1);
        absQuant[j * 32 + 10] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 11] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 12] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 13] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 14] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 15] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);

        // Get last bit in 16-23 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 16] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 17] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 18] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 19] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 20] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 21] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 22] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 23] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);

        // Get last bit in 24-31 abs quant from global memory, padding part.
        uchar_buffer = CmpDataIn[cmp_byte_ofs++];
        absQuant[j * 32 + 24] |= ((uchar_buffer >> 7) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 25] |= ((uchar_buffer >> 6) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 26] |= ((uchar_buffer >> 5) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 27] |= ((uchar_buffer >> 4) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 28] |= ((uchar_buffer >> 3) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 29] |= ((uchar_buffer >> 2) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 30] |= ((uchar_buffer >> 1) & 0x00000001)
                                 << (fixed_rate[j] - 1);
        absQuant[j * 32 + 31] |= ((uchar_buffer >> 0) & 0x00000001)
                                 << (fixed_rate[j] - 1);
      }
      //  Decompress and sum with predQuant.
      maxQuan2 = 0;
      maxQuant = 0;
      float4 tmp_buffer;
      int current_quant = 0;
      int prevQuant = 0;
#pragma unroll 8
      for (int i = 0; i < 32; i += 4) {
        tmp_buffer = reinterpret_cast<float4 *>(
            localChunk)[(base_block_start_idx + i) / 4];
        current_quant = quantization(tmp_buffer.x, recipPrecision);
        lorenQuant = current_quant - prevQuant;
        lorenQuant = ((sign_flag & (1 << (31 - i))) ? absQuant[j * 32 + i] * -1
                                                    : absQuant[j * 32 + i]) +
                     lorenQuant;
        prevQuant = current_quant;
        sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - i);
        absQuant[j * 32 + i] = abs(lorenQuant);
        maxQuant = max(maxQuant, absQuant[j * 32 + i]);
        if (i)
          maxQuan2 = max(maxQuan2, absQuant[j * 32 + i]);

        current_quant = quantization(tmp_buffer.y, recipPrecision);
        lorenQuant = current_quant - prevQuant;
        lorenQuant =
            ((sign_flag & (1 << (31 - (i + 1))) ? absQuant[j * 32 + i + 1] * -1
                                                : absQuant[j * 32 + i + 1])) +
            lorenQuant;
        prevQuant = current_quant;
        sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - (i + 1));
        absQuant[j * 32 + i + 1] = abs(lorenQuant);
        maxQuant = max(maxQuant, absQuant[j * 32 + i + 1]);
        maxQuan2 = max(maxQuan2, absQuant[j * 32 + i + 1]);

        current_quant = quantization(tmp_buffer.z, recipPrecision);
        lorenQuant = current_quant - prevQuant;
        lorenQuant =
            ((sign_flag & (1 << (31 - (i + 2))) ? absQuant[j * 32 + i + 2] * -1
                                                : absQuant[j * 32 + i + 2])) +
            lorenQuant;
        prevQuant = current_quant;
        sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - (i + 2));
        absQuant[j * 32 + i + 2] = abs(lorenQuant);
        maxQuant = max(maxQuant, absQuant[j * 32 + i + 2]);
        maxQuan2 = max(maxQuan2, absQuant[j * 32 + i + 2]);

        current_quant = quantization(tmp_buffer.w, recipPrecision);
        lorenQuant = current_quant - prevQuant;
        lorenQuant =
            ((sign_flag & (1 << (31 - (i + 3))) ? absQuant[j * 32 + i + 3] * -1
                                                : absQuant[j * 32 + i + 3])) +
            lorenQuant;
        prevQuant = current_quant;
        sign_flag_cmp[j] |= (lorenQuant < 0) << (31 - (i + 3));
        absQuant[j * 32 + i + 3] = abs(lorenQuant);
        maxQuant = max(maxQuant, absQuant[j * 32 + i + 3]);
        maxQuan2 = max(maxQuan2, absQuant[j * 32 + i + 3]);
      }
      outlier = absQuant[j * 32];
    }

    int fr1 = get_bit_num(maxQuant);
    int fr2 = get_bit_num(maxQuan2);
    outlier = (get_bit_num(outlier) + 7) / 8;
    int temp_rate = 0;
    int temp_ofs1 = fr1 ? 4 + fr1 * 4 : 0;
    int temp_ofs2 = fr2 ? 4 + fr2 * 4 + outlier : 4 + outlier;
    if (temp_ofs1 <= temp_ofs2) {
      thread_ofs2 += temp_ofs1;
      temp_rate = fr1;
    } else {
      thread_ofs2 += temp_ofs2;
      temp_rate = fr2 | 0x80 | ((outlier - 1) << 5);
    }
    fixed_rate_cmp[j] = temp_rate;
    CmpDataOut[(base_block_start_idx / 32)] =
        (unsigned char)fixed_rate_cmp[j]; // ERROR
    __syncthreads();
    // Index updating across different iterations.
    cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
  }

#pragma unroll 5
  for (int i = 1; i < 32; i <<= 1) {
    int tmp = __shfl_up_sync(0xffffffff, thread_ofs2, i);
    if (lane >= i)
      thread_ofs2 += tmp;
  }
  __syncthreads();

  // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
  if (lane == 31) {
    locOffsetOut[warp + 1] = thread_ofs2;
    __threadfence();
    if (warp == 0) {
      flag_cmp[0] = 2;
      __threadfence();
      flag_cmp[1] = 1;
      __threadfence();
    } else {
      flag_cmp[warp + 1] = 1;
      __threadfence();
    }
  }
  __syncthreads();

  // Global-level prefix-sum (exclusive).
  if (warp > 0) {
    if (!lane) {
      // Decoupled look-back
      int lookback = warp;
      int loc_excl_sum = 0;
      while (lookback > 0) {
        int status;
        // Local sum not end.
        do {
          status = flag_cmp[lookback];
          __threadfence();
        } while (status == 0);
        // Lookback end.
        if (status == 2) {
          loc_excl_sum += CmpOffsetOut[lookback];
          __threadfence();
          break;
        }
        // Continues lookback.
        if (status == 1)
          loc_excl_sum += locOffsetOut[lookback];
        lookback--;
        __threadfence();
      }
      excl_sum = loc_excl_sum;
    }
    __syncthreads();
  }

  if (warp > 0) {
    // Update global flag.
    if (!lane)
      CmpOffsetOut[warp] = excl_sum;
    __threadfence();
    if (!lane)
      flag_cmp[warp] = 2;
    __threadfence();
  }
  __syncthreads();

  // Assigning compression bytes by given prefix-sum results.
  if (!lane)
    base_idx = excl_sum + rate_ofs;
  __syncthreads();
  base_cmp_byte_ofs = base_idx;
  tmp_byte_ofs = 0;
  cur_byte_ofs = 0;
  for (int j = 0; j < block_num; j++) {
    // Initialization, guiding encoding process.
    int encoding_selection = fixed_rate_cmp[j] >> 7;
    int outlier_byte_num = ((fixed_rate_cmp[j] & 0x60) >> 5) + 1;
    fixed_rate_cmp[j] &= 0x1f;
    int chunk_idx_start = j * 32;

    // Restore index for j-th iteration.
    if (!encoding_selection)
      tmp_byte_ofs = (fixed_rate_cmp[j]) ? (4 + fixed_rate_cmp[j] * 4) : 0;
    else
      tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate_cmp[j] * 4;
#pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
      int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
      if (lane >= i)
        tmp_byte_ofs += tmp;
    }
    unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
    if (!lane)
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
    else
      cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

    // If outlier encoding, storing outliers here.
    if (encoding_selection) {
      for (int i = 0; i < outlier_byte_num; i++) {
        CmpDataOut[cmp_byte_ofs++] =
            (unsigned char)(absQuant[chunk_idx_start] & 0xff);
        absQuant[chunk_idx_start] >>= 8;
      }

      // Corner case: all data points except outliers are 0.
      if (!fixed_rate_cmp[j]) {
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 16);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 8);
        CmpDataOut[cmp_byte_ofs++] = 0xff & sign_flag_cmp[j];
      }
    }

    // Operation for each block, if zero block then do nothing.
    if (fixed_rate_cmp[j]) {
      // Padding vector operation for outlier encoding.
      int vec_ofs = cmp_byte_ofs % 4;
      if (vec_ofs == 0) {
        // Assign sign information for one block.
        tmp_char.x = 0xff & (sign_flag_cmp[j] >> 24);
        tmp_char.y = 0xff & (sign_flag_cmp[j] >> 16);
        tmp_char.z = 0xff & (sign_flag_cmp[j] >> 8);
        tmp_char.w = 0xff & sign_flag_cmp[j];
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j]; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.x = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.x = tmp_char.x |
                       (((absQuant[chunk_idx_start + 0] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 1] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 2] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 3] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 4] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 5] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 6] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 7] & mask) >> i) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.y = (((absQuant[chunk_idx_start + 8] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 9] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 10] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 11] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 12] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 13] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 14] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 15] & mask) >> i) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.z = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.w = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
          mask <<= 1;
        }
      } else if (vec_ofs == 1) {
        // Assign sign information for one block, padding part.
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 16);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 8);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & sign_flag_cmp[j];
        if (!encoding_selection)
          tmp_char.y = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.y = tmp_char.y | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        tmp_char.z = ((absQuant[chunk_idx_start + 8] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 9] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 10] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 11] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 12] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 13] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 14] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 15] & 1) << 0);
        tmp_char.w = ((absQuant[chunk_idx_start + 16] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 17] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 18] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 19] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 20] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 21] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 22] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 23] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.x = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.y = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.y =
              tmp_char.y |
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.z =
              (((absQuant[chunk_idx_start + 8] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 9] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 10] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 11] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 12] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 13] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 14] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 15] & mask) >> (i + 1)) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.w =
              (((absQuant[chunk_idx_start + 16] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 17] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 18] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 19] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 20] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 21] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 22] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 23] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
      } else if (vec_ofs == 2) {
        // Assign sign information for one block, padding part.
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 16);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & (sign_flag_cmp[j] >> 8);
        tmp_char.y = 0xff & sign_flag_cmp[j];
        if (!encoding_selection)
          tmp_char.z = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.z = tmp_char.z | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        tmp_char.w = ((absQuant[chunk_idx_start + 8] & 1) << 7) |
                     ((absQuant[chunk_idx_start + 9] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 10] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 11] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 12] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 13] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 14] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 15] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.x = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.y = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.z = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.z =
              tmp_char.z |
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.w =
              (((absQuant[chunk_idx_start + 8] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 9] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 10] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 11] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 12] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 13] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 14] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 15] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 16] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 17] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 18] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 19] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 20] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 21] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 22] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 23] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
      } else {
        // Assign sign information for one block, padding part.
        CmpDataOut[cmp_byte_ofs++] = 0xff & (sign_flag_cmp[j] >> 24);

        // Assign sign information and block bit-shuffle, padding for
        // vectorization.
        tmp_char.x = 0;
        tmp_char.y = 0;
        tmp_char.z = 0;
        tmp_char.w = 0;
        tmp_char.x = 0xff & (sign_flag_cmp[j] >> 16);
        tmp_char.y = 0xff & (sign_flag_cmp[j] >> 8);
        tmp_char.z = 0xff & sign_flag_cmp[j];
        if (!encoding_selection)
          tmp_char.w = ((absQuant[chunk_idx_start] & 1) << 7);
        tmp_char.w = tmp_char.w | ((absQuant[chunk_idx_start + 1] & 1) << 6) |
                     ((absQuant[chunk_idx_start + 2] & 1) << 5) |
                     ((absQuant[chunk_idx_start + 3] & 1) << 4) |
                     ((absQuant[chunk_idx_start + 4] & 1) << 3) |
                     ((absQuant[chunk_idx_start + 5] & 1) << 2) |
                     ((absQuant[chunk_idx_start + 6] & 1) << 1) |
                     ((absQuant[chunk_idx_start + 7] & 1) << 0);
        reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
        cmp_byte_ofs += 4;

        // Assign quant bit information for one block by bit-shuffle.
        int mask = 1;
        for (int i = 0; i < fixed_rate_cmp[j] - 1; i++) {
          // Initialization.
          tmp_char.x = 0;
          tmp_char.y = 0;
          tmp_char.z = 0;
          tmp_char.w = 0;

          // Get ith bit in 8~15 quant, and store to tmp_char.y
          tmp_char.x = (((absQuant[chunk_idx_start + 8] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 9] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 10] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 11] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 12] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 13] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 14] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 15] & mask) >> i) << 0);

          // Get ith bit in 16~23 quant, and store to tmp_char.z
          tmp_char.y = (((absQuant[chunk_idx_start + 16] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 17] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 18] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 19] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 20] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 21] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 22] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 23] & mask) >> i) << 0);

          // Get ith bit in 24-31 quant, and store to tmp_char.w
          tmp_char.z = (((absQuant[chunk_idx_start + 24] & mask) >> i) << 7) |
                       (((absQuant[chunk_idx_start + 25] & mask) >> i) << 6) |
                       (((absQuant[chunk_idx_start + 26] & mask) >> i) << 5) |
                       (((absQuant[chunk_idx_start + 27] & mask) >> i) << 4) |
                       (((absQuant[chunk_idx_start + 28] & mask) >> i) << 3) |
                       (((absQuant[chunk_idx_start + 29] & mask) >> i) << 2) |
                       (((absQuant[chunk_idx_start + 30] & mask) >> i) << 1) |
                       (((absQuant[chunk_idx_start + 31] & mask) >> i) << 0);
          mask <<= 1;

          // Get ith bit in 0~7 quant, and store to tmp_char.x
          // If using outlier fixed-length encoding, then first element is set
          // as 0.
          if (!encoding_selection)
            tmp_char.w = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
          tmp_char.w =
              (((absQuant[chunk_idx_start + 0] & mask) >> (i + 1)) << 7) |
              (((absQuant[chunk_idx_start + 1] & mask) >> (i + 1)) << 6) |
              (((absQuant[chunk_idx_start + 2] & mask) >> (i + 1)) << 5) |
              (((absQuant[chunk_idx_start + 3] & mask) >> (i + 1)) << 4) |
              (((absQuant[chunk_idx_start + 4] & mask) >> (i + 1)) << 3) |
              (((absQuant[chunk_idx_start + 5] & mask) >> (i + 1)) << 2) |
              (((absQuant[chunk_idx_start + 6] & mask) >> (i + 1)) << 1) |
              (((absQuant[chunk_idx_start + 7] & mask) >> (i + 1)) << 0);

          // Move data to global memory via a vectorized pattern.
          reinterpret_cast<uchar4 *>(CmpDataOut)[cmp_byte_ofs / 4] = tmp_char;
          cmp_byte_ofs += 4;
        }

        // Assign block bit-shuffle, padding part.
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 8] & mask) >> (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 9] & mask) >> (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 10] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 11] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 12] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 13] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 14] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 15] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 16] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 17] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 18] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 19] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 20] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 21] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 22] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 23] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
        CmpDataOut[cmp_byte_ofs++] =
            (((absQuant[chunk_idx_start + 24] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 7) |
            (((absQuant[chunk_idx_start + 25] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 6) |
            (((absQuant[chunk_idx_start + 26] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 5) |
            (((absQuant[chunk_idx_start + 27] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 4) |
            (((absQuant[chunk_idx_start + 28] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 3) |
            (((absQuant[chunk_idx_start + 29] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 2) |
            (((absQuant[chunk_idx_start + 30] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 1) |
            (((absQuant[chunk_idx_start + 31] & mask) >>
              (fixed_rate_cmp[j] - 1))
             << 0);
      }
    }

    cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
  }
}