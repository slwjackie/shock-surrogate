/* Compose native eight-step updates over a batch of independent trajectories.
   Each block reads the same method's previous result. */
#include <stdlib.h>
#include <stdint.h>

int sw_macro8_batch(int, int, const double *, double *, int, double, int *);

int sw_macro8_trajectory_batch(int count, int n, int blocks, const double *q,
        double *out, int method, double threshold, int64_t *accepted) {
    if (count < 1 || n < 2 || blocks < 1 || !q || !out || !accepted || q == out ||
        (size_t)count > SIZE_MAX / (2 * sizeof(double)) / (size_t)n ||
        (size_t)count > SIZE_MAX / (3 * sizeof(int))) return -1;
    size_t doubles = (size_t)n * 2u * (size_t)count;
    double *intermediate = malloc(doubles * sizeof(double));
    int *stats = malloc(3u * (size_t)count * sizeof(int));
    if (!intermediate || !stats) {
        free(intermediate);
        free(stats);
        return -6;
    }
    *accepted = 0;
    int status = 0;
    const double *input = q;
    for (int k = 0; k < blocks; ++k) {
        double *next = ((blocks-k) % 2 == 1) ? out : intermediate;
        status = sw_macro8_batch(count, n, input, next, method, threshold, stats);
        if (status) break;
        for (int j = 0; j < count; ++j)
            *accepted += stats[3u*(size_t)j] == 1;
        input = next;
    }
    free(intermediate);
    free(stats);
    return status;
}
