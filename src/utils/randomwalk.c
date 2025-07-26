#include <stdio.h>
#include <stdlib.h>
#include <time.h>

inline float random_step(float max_step) {
  return ((float)rand() / RAND_MAX) * 2 * max_step - max_step;
}

int main(int argc, char *argv[]) {
  int n = 1000000;
  float max_step = 0.5f;
  float current = 0.0f;
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <file_name>\n", argv[0]);
    return 1;
  }
  char *file_name = argv[1];

  srand(time(NULL));

  FILE *file = fopen(file_name, "w");
  if (file == NULL) {
    fprintf(stderr, "Error opening file\n");
    return 1;
  }
  for (int i = 0; i < n; i++) {
    fprintf(file, "%f\n", current);
    current += random_step(max_step);
  }
  fclose(file);
  return 0;
}
