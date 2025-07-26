#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int next_poweroftwo(int n) {
  int i = 1;
  while (i < n) {
    i <<= 1;
  }
  return i;
}

ptrdiff_t datatype_span(MPI_Datatype dtype, size_t count, ptrdiff_t *gap) {
  int ret;
  ptrdiff_t lb, extent, true_lb, true_extent;

  ret = MPI_Type_get_extent(dtype, &lb, &extent);
  if (MPI_SUCCESS != ret) {
    return -1;
  }
  ret = MPI_Type_get_true_extent(dtype, &true_lb, &true_extent);
  if (MPI_SUCCESS != ret) {
    return -1;
  }

  *gap = true_lb - lb;
  return (ptrdiff_t)(count * extent + *gap);
}

int sizeof_datatype(MPI_Datatype dtype) {
  int size;
  MPI_Type_size(dtype, &size);
  return size;
}

/**
 * This macro gives a generic way to compute the well distributed block counts
 * when the count and number of blocks are fixed.
 * Macro returns "early-block" count, "late-block" count, and "split-index"
 * which is the block at which we switch from "early-block" count to
 * the "late-block" count.
 * count = split_index * early_block_count +
 *         (block_count - split_index) * late_block_count
 * We do not perform ANY error checks - make sure that the input values
 * make sense (eg. count > num_blocks).
 */
#define COLL_BASE_COMPUTE_BLOCKCOUNT(COUNT, NUM_BLOCKS, SPLIT_INDEX,           \
                                     EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT)      \
  EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;                   \
  SPLIT_INDEX = COUNT % NUM_BLOCKS;                                            \
  if (0 != SPLIT_INDEX) {                                                      \
    EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                                 \
  }

int allreduce_recursivedoubling(const void *sbuf, void *rbuf, size_t count,
                                MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  int ret, line, rank, size, adjsize, remote, distance;
  void *ret_;
  int newrank, newremote, extra_ranks;
  char *tmpsend = NULL, *tmprecv = NULL, *inplacebuf_free = NULL, *inplacebuf;
  ptrdiff_t span, gap = 0;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /* Special case for size == 1 */
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret_ = memcpy((char *)sbuf, (char *)rbuf, count * sizeof_datatype(dtype));
      if (ret_ == NULL) {
        line = __LINE__;
        goto error_hndl;
      }
    }
    return MPI_SUCCESS;
  }

  /* Allocate and initialize temporary send buffer */
  span = datatype_span(dtype, count, &gap);

  inplacebuf_free = (char *)malloc(span);
  if (NULL == inplacebuf_free) {
    line = __LINE__;
    goto error_hndl;
  }
  inplacebuf = inplacebuf_free - gap;

  if (MPI_IN_PLACE == sbuf) {
    ret_ = memcpy((char *)rbuf, inplacebuf, count * sizeof_datatype(dtype));
    if (ret_ == NULL) {
      line = __LINE__;
      goto error_hndl;
    }
  } else {
    ret_ = memcpy((char *)sbuf, inplacebuf, count * sizeof_datatype(dtype));
    if (ret_ == NULL) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  tmpsend = (char *)inplacebuf;
  tmprecv = (char *)rbuf;

  /* Determine nearest power of two less than or equal to size */
  adjsize = next_poweroftwo(size) >> 1;

  /* Handle non-power-of-two case:
  - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
  sets new rank to -1.
  - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
  apply appropriate operation, and set new rank to rank/2
  - Everyone else sets rank to rank - extra_ranks
  */
  extra_ranks = size - adjsize;
  if (rank < (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if (MPI_SUCCESS != ret) {
        line = __LINE__;
        goto error_hndl;
      }
      newrank = -1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm,
                     MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) {
        line = __LINE__;
        goto error_hndl;
      }
      /* tmpsend = tmprecv (op) tmpsend */
      // reduction((int64_t *) tmprecv, (int64_t *) tmpsend, count);
      MPI_Reduce_local((char *)tmprecv, (char *)tmpsend, count, dtype, op);
      newrank = rank >> 1;
    }
  } else {
    newrank = rank - extra_ranks;
  }

  /* Communication/Computation loop
  - Exchange message with remote node.
  - Perform appropriate operation taking in account order of operations:
  result = value (op) result
  */
  for (distance = 0x1; distance < adjsize; distance <<= 1) {
    if (newrank < 0)
      break;
    /* Determine remote node */
    newremote = newrank ^ distance;
    remote = (newremote < extra_ranks) ? (newremote * 2 + 1)
                                       : (newremote + extra_ranks);

    /* Exchange the data */
    ret = MPI_Sendrecv(tmpsend, count, dtype, remote, 0, tmprecv, count, dtype,
                       remote, 0, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }

    // reduction((int64_t *) tmprecv, (int64_t *) tmpsend, count);
    MPI_Reduce_local((char *)tmprecv, (char *)tmpsend, count, dtype, op);
  }

  /* Handle non-power-of-two case:
  - Odd ranks less than 2 * extra_ranks send result from tmpsend to
  (rank - 1)
  - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
  */
  if (rank < (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret =
          MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) {
        line = __LINE__;
        goto error_hndl;
      }
      tmpsend = (char *)rbuf;
    } else {
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
      if (MPI_SUCCESS != ret) {
        line = __LINE__;
        goto error_hndl;
      }
    }
  }

  /* Ensure that the final result is in rbuf */
  if (tmpsend != rbuf) {
    ret_ = memcpy(tmpsend, (char *)rbuf, count * sizeof_datatype(dtype));
    if (ret_ == NULL) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  if (NULL != inplacebuf_free)
    free(inplacebuf_free);
  return MPI_SUCCESS;

error_hndl:
  fprintf(stderr, "\n%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank,
          ret);
  (void)line; // silence compiler warning
  if (NULL != inplacebuf_free)
    free(inplacebuf_free);
  return ret;
}

int allreduce_ring(const void *sbuf, void *rbuf, size_t count,
                   MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
  void *ret_;
  int early_segcount, late_segcount, split_rank, max_segcount;
  char *tmpsend = NULL, *tmprecv = NULL, *inbuf[2] = {NULL, NULL};
  ptrdiff_t true_lb, true_extent, lb, extent;
  ptrdiff_t block_offset, max_real_segsize;
  MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  ret = MPI_Comm_rank(comm, &rank); // get rank
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }
  ret = MPI_Comm_size(comm, &size); // get size of comm
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  // if(rank == 0) {
  //   printf("4: RING\n");
  //   fflush(stdout);
  // }

  /* Special case for size == 1 */
  if (1 == size) { // if only one rank, no need to do anything
    if (MPI_IN_PLACE != sbuf) {
      ret_ = memcpy((char *)sbuf, (char *)rbuf, count * sizeof_datatype(dtype));
      if (ret_ < 0) {
        line = __LINE__;
        goto error_hndl;
      }
    }
    return MPI_SUCCESS;
  }

  /* Special case for count less than size - use recursive doubling */
  if (count < (size_t)size) { // if count (elements to send) is less than size,
                              // use recursive doubling
    return (allreduce_recursivedoubling(sbuf, rbuf, count, dtype, op, comm));
  }

  /* Allocate and initialize temporary buffers */
  ret = MPI_Type_get_extent(dtype, &lb, &extent); // get extent of datatype
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }
  ret = MPI_Type_get_true_extent(dtype, &true_lb,
                                 &true_extent); // get true extent
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  /* Determine the number of elements per block and corresponding
  block sizes.
  The blocks are divided into "early" and "late" ones:
  blocks 0 .. (split_rank - 1) are "early" and
  blocks (split_rank) .. (size - 1) are "late".
  Early blocks are at most 1 element larger than the late ones.
  */
  // compute the block with +1 element respect to the other
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, size, split_rank, early_segcount,
                               late_segcount);
  max_segcount = early_segcount;
  max_real_segsize = true_extent + (max_segcount - 1) * extent;

  inbuf[0] = (char *)malloc(max_real_segsize);
  if (NULL == inbuf[0]) {
    ret = -1;
    line = __LINE__;
    goto error_hndl;
  }
  if (size > 2) {
    inbuf[1] = (char *)malloc(max_real_segsize);
    if (NULL == inbuf[1]) {
      ret = -1;
      line = __LINE__;
      goto error_hndl;
    }
  }

  /* Handle MPI_IN_PLACE */
  if (MPI_IN_PLACE != sbuf) {
    ret_ = memcpy((char *)sbuf, (char *)rbuf, count * sizeof_datatype(dtype));
    if (ret_ == NULL) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  /* Computation loop */

  /*
  For each of the remote nodes:
  - post irecv for block (r-1)
  - send block (r)
  - in loop for every step k = 2 .. n
  - post irecv for block (r + n - k) % n
  - wait on block (r + n - k + 1) % n to arrive
  - compute on block (r + n - k + 1) % n
  - send block (r + n - k + 1) % n
  - wait on block (r + 1)
  - compute on block (r + 1)
  - send block (r + 1) to rank (r + 1)
  Note that we must be careful when computing the beginning of buffers and
  for send operations and computation we must compute the exact block size.
  */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;

  inbi = 0;
  /* Initialize first receive from the neighbor on the left */
  ret = MPI_Irecv(inbuf[inbi], max_segcount, dtype, recv_from, 0, comm,
                  &reqs[inbi]);
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }
  /* Send first block (my block) to the neighbor on the right */
  block_offset =
      ((rank < split_rank)
           ? ((ptrdiff_t)rank * (ptrdiff_t)early_segcount)
           : ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
  block_count = ((rank < split_rank) ? early_segcount : late_segcount);
  tmpsend = ((char *)rbuf) + block_offset * extent;
  ret = MPI_Send(tmpsend, block_count, dtype, send_to, 0, comm);
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  for (k = 2; k < size; k++) {
    const int prevblock = (rank + size - k + 1) % size;

    inbi = inbi ^ 0x1;

    /* Post irecv for the current block */
    ret = MPI_Irecv(inbuf[inbi], max_segcount, dtype, recv_from, 0, comm,
                    &reqs[inbi]);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }

    /* Wait on previous block to arrive */
    ret = MPI_Wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }

    /* Apply operation on previous block: result goes to rbuf
    rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
    */
    block_offset = ((prevblock < split_rank)
                        ? ((ptrdiff_t)prevblock * early_segcount)
                        : ((ptrdiff_t)prevblock * late_segcount + split_rank));
    block_count = ((prevblock < split_rank) ? early_segcount : late_segcount);
    tmprecv = ((char *)rbuf) + (ptrdiff_t)block_offset * extent;
    MPI_Reduce_local(inbuf[inbi ^ 0x1], tmprecv, block_count, dtype, op);

    /* send previous block to send_to */
    ret = MPI_Send(tmprecv, block_count, dtype, send_to, 0, comm);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  /* Wait on the last block to arrive */
  ret = MPI_Wait(&reqs[inbi], MPI_STATUS_IGNORE);
  if (MPI_SUCCESS != ret) {
    line = __LINE__;
    goto error_hndl;
  }

  /* Apply operation on the last block (from neighbor (rank + 1)
  rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
  recv_from = (rank + 1) % size;
  block_offset = ((recv_from < split_rank)
                      ? ((ptrdiff_t)recv_from * early_segcount)
                      : ((ptrdiff_t)recv_from * late_segcount + split_rank));
  block_count = ((recv_from < split_rank) ? early_segcount : late_segcount);
  tmprecv = ((char *)rbuf) + (ptrdiff_t)block_offset * extent;
  MPI_Reduce_local(inbuf[inbi], tmprecv, block_count, dtype, op);

  /* Distribution loop - variation of ring allgather */
  send_to = (rank + 1) % size;
  recv_from = (rank + size - 1) % size;
  for (k = 0; k < size - 1; k++) {
    const int recv_data_from = (rank + size - k) % size;
    const int send_data_from = (rank + 1 + size - k) % size;
    const int send_block_offset =
        ((send_data_from < split_rank)
             ? ((ptrdiff_t)send_data_from * early_segcount)
             : ((ptrdiff_t)send_data_from * late_segcount + split_rank));
    const int recv_block_offset =
        ((recv_data_from < split_rank)
             ? ((ptrdiff_t)recv_data_from * early_segcount)
             : ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
    block_count =
        ((send_data_from < split_rank) ? early_segcount : late_segcount);

    tmprecv = (char *)rbuf + (ptrdiff_t)recv_block_offset * extent;
    tmpsend = (char *)rbuf + (ptrdiff_t)send_block_offset * extent;

    ret = MPI_Sendrecv(tmpsend, block_count, dtype, send_to, 0, tmprecv,
                       max_segcount, dtype, recv_from, 0, comm,
                       MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) {
      line = __LINE__;
      goto error_hndl;
    }
  }

  if (NULL != inbuf[0])
    free(inbuf[0]);
  if (NULL != inbuf[1])
    free(inbuf[1]);

  return MPI_SUCCESS;

error_hndl:
  fprintf(stderr, "\n%s:%4d\tRank %d Error occurred %d\n\n", __FILE__, line,
          rank, ret);
  MPI_Request_free(&reqs[0]);
  MPI_Request_free(&reqs[1]);
  (void)line; // silence compiler warning
  if (NULL != inbuf[0])
    free(inbuf[0]);
  if (NULL != inbuf[1])
    free(inbuf[1]);
  return ret;
}