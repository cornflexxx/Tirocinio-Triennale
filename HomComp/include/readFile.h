#include <stdio.h>
#include <stdlib.h>

float *read_binary_floats(const char *filename, size_t *out_size) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Cannot open file");
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  long filesize = ftell(file);
  rewind(file);

  size_t count = filesize / sizeof(float);
  float *buffer = (float *)malloc(filesize);
  if (!buffer) {
    fclose(file);
    perror("Memory allocation failed");
    return NULL;
  }

  size_t read = fread(buffer, sizeof(float), count, file);
  if (read != count) {
    perror("Error reading file");
    free(buffer);
    fclose(file);
    return NULL;
  }

  fclose(file);
  *out_size = count;
  return buffer;
}
