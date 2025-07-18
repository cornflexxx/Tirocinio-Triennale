#ifndef READFILEF32_H
#define READFILEF32_H

#include <stdio.h>
#include <stdlib.h>

float *readFilef32(const char *filename, size_t *nbEle) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    perror("Errore apertura file");
    *nbEle = 0;
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  long filesize = ftell(fp);
  rewind(fp);

  if (filesize % sizeof(float) != 0) {
    fprintf(stderr, "Errore: dimensione file non multipla di 4 byte\n");
    fclose(fp);
    *nbEle = 0;
    return NULL;
  }

  *nbEle = filesize / sizeof(float);

  float *data = (float *)malloc(filesize);
  if (!data) {
    fprintf(stderr, "Errore allocazione memoria\n");
    fclose(fp);
    *nbEle = 0;
    return NULL;
  }

  size_t re = fread(data, sizeof(float), *nbEle, fp);
  if (re != *nbEle) {
    fprintf(stderr, "Attenzione: letti solo %zu valori su %zu\n", letti,
            *nbEle);
  }

  fclose(fp);
  return data;
}

#endif