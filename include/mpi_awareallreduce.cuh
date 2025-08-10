#pragma once
#include <mpi.h>

int allreduce_ring_comprs_hom_sum(const float *d_sbuf, float *d_rbuf,
                                  size_t count, MPI_Comm comm, float eb);
int allreduce_ring_comprs_hom_sum_F(const float *d_sbuf, float *d_rbuf,
                                    size_t count, MPI_Comm comm, float eb);
int allreduce_ring_comprs_hom_sum_seg(const float *d_sbuf, float *d_rbuf,
                                      size_t count, MPI_Comm comm, float eb);
int allreduce_ring_comprs_hom_sum_F_seg(const float *d_sbuf, float *d_rbuf,
                                        size_t count, MPI_Comm comm, float eb);
int allreduce_ring_comprs_hom_sum_F_opt(const float *d_sbuf, float *d_rbuf,
                                        size_t count, MPI_Comm comm, float eb);
int mixed_compressed_allreduce(float *d_sbuf, float *d_rbuf, size_t count,
                               MPI_Comm comm, float eb);
int allreduce_ring_gpu(const float *d_sbuf, float *d_rbuf, size_t count,
                       MPI_Op op, MPI_Comm comm);