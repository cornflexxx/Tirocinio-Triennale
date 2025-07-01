#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *log_file = NULL;

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (!log_file) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    char filename[64];
    snprintf(filename, sizeof(filename), "allreduce_log_rank%d.txt", rank);
    log_file = fopen(filename, "a");
    if (!log_file) {
      perror("fopen");
      exit(EXIT_FAILURE);
    }
  }

  fprintf(log_file, "MPI_Allreduce called with count = %d\n", count);

  int type_size;
  MPI_Type_size(datatype, &type_size);

  fprintf(log_file, "Input values: ");
  for (int i = 0; i < count; i++) {
    if (datatype == MPI_FLOAT)
      fprintf(log_file, "%f ", ((float *)sendbuf)[i);
    else
      break;
  }
  fprintf(log_file, "\n");

  int ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

  fprintf(log_file, "Output values: ");
  for (int i = 0; i < count; i++) {
    if (datatype == MPI_FLOAT)
      fprintf(log_file, "%f ", ((float *)recvbuf)[i]);
    else
      break;
  }

  fprintf(log_file, "\n---\n");
  fflush(log_file);

  return ret;
}
