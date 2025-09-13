#ifndef GSZ_INCLUDE_GSZ_ENTRY_H
#define GSZ_INCLUDE_GSZ_ENTRY_H

void GSZ_compress_hostptr(float *oriData, unsigned char *cmpBytes, size_t nbEle,
                          size_t *cmpSize, float errorBound);
void GSZ_decompress_hostptr(float *decData, unsigned char *cmpBytes,
                            size_t nbEle, size_t cmpSize, float errorBound);
void GSZ_compress_deviceptr_plain(float *d_oriData, unsigned char *d_cmpBytes,
                                  size_t nbEle, size_t *cmpSize,
                                  float errorBound, cudaStream_t stream = 0);
void GSZ_decompress_deviceptr_plain(float *d_decData, unsigned char *d_cmpBytes,
                                    size_t nbEle, size_t cmpSize,
                                    float errorBound, cudaStream_t stream = 0);
void GSZ_compress_deviceptr_outlier(float *d_oriData, unsigned char *d_cmpBytes,
                                    size_t nbEle, size_t *cmpSize,
                                    float errorBound, cudaStream_t stream = 0);
void GSZ_decompress_deviceptr_outlier(float *d_decData,
                                      unsigned char *d_cmpBytes, size_t nbEle,
                                      size_t cmpSize, float errorBound,
                                      cudaStream_t stream = 0);
void GSZ_compress_deviceptr_outlier_vec(float *d_oriData,
                                        unsigned char *d_cmpBytes, size_t nbEle,
                                        size_t *cmpSize, float errorBound,
                                        cudaStream_t stream = 0);
void GSZ_decompress_deviceptr_outlier_vec(float *d_decData,
                                          unsigned char *d_cmpBytes,
                                          size_t nbEle, size_t cmpSize,
                                          float errorBound,
                                          cudaStream_t stream = 0);
void homomorphic_sum(unsigned char *d_cmpBytesIn, int *d_quantPredLoc,
                     unsigned char *d_cmpByteOut, size_t nbEle,
                     float errorBound, size_t *cmpSize, size_t cmpSizeCmpBlock,
                     cudaStream_t stream = 0);
void homomorphic_sum_F(unsigned char *d_cmpBytesIn, float *d_localChunk,
                       unsigned char *d_cmpByteOut, size_t nbEle,
                       float errorBound, size_t *cmpSize,
                       size_t cmpSizeCmpBlock, cudaStream_t stream = 0);
#endif // GSZ_INCLUDE_GSZ_ENTRY_H