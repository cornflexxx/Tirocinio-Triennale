#pragma once
#include <mpi.h>
int cpuCopy_allreduce_ring_comprs_hom_sum(const float *d_sbuf, float *d_rbuf,
                                          size_t count, MPI_Comm comm,
                                          float eb);
int cpuCopy_allreduce_ring_comprs_hom_sum_F(const float *d_sbuf, float *d_rbuf,
                                            size_t count, MPI_Comm comm,
                                            float eb);

int cpuCopy_allreduce_ring_comprs_hom_sum_F_seg(const float *d_sbuf,
                                                float *d_rbuf, size_t count,
                                                MPI_Comm comm, float eb);
int cpuCopy_allreduce_ring_comprs_hom_sum_seg(const float *d_sbuf,
                                              float *d_rbuf, size_t count,
                                              MPI_Comm comm, float eb);