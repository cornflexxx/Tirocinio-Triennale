#include <math.h>
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

int main(int argc, char **argv) {
  size_t cIn;
  int nproc = atoi(argv[1]);
  float *smoothin = read_data("smooth.in", &cIn);
  float *smoothout = read_data("smooth.out", &cIn);
  float eb = 0.0001;
  int nb = 0;
  for (int i = 0; i < cIn; i++) {
    if (fabs(smoothin[i] * nproc - smoothout[i]) >
        2 * eb * (nproc + ((float)nproc / 10))) {
      nb++;
      printf("%f , %f\n", smoothin[i], smoothout[i]);
    }
  }
  if (!nb)
    printf("\033[0;32mPass error check!\033[0m\n");
  else
    printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", nb);
}