/* Compute-only FP64 control. Model arrays in weights.h are generated from
   the checked-in frozen flux. Each SIMD lane is an independent face. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "weights.h"

enum { N = 128, WIDTH = 16, LANES = 8 };

static inline double relu(double x) { return x > 0.0 ? x : 0.0; }

static void neural_eight(const double *state, int first, double *flux)
{
    double h1[WIDTH][LANES], h2[WIDTH][LANES];

    for (int k = 0; k < WIDTH; k++) {
        for (int l = 0; l < LANES; l++) {
            int i = first + l;
            double pre = b0[k] + state[i] * w0[k][0];
            pre += state[(i+1) % N] * w0[k][1];
            h1[k][l] = relu(pre);
        }
    }

    for (int k = 0; k < WIDTH; k++) {
        double acc[LANES];
        for (int l = 0; l < LANES; l++)
            acc[l] = b1[k];
        for (int j = 0; j < WIDTH; j++) {
            for (int l = 0; l < LANES; l++)
                acc[l] += h1[j][l] * w1[k][j];
        }
        for (int l = 0; l < LANES; l++)
            h2[k][l] = relu(acc[l]);
    }

    for (int l = 0; l < LANES; l++)
        flux[first+l] = b2[0];
    for (int j = 0; j < WIDTH; j++) {
        for (int l = 0; l < LANES; l++)
            flux[first+l] += h2[j][l] * w2[0][j];
    }
}

static double godunov(double left, double right)
{
    double left_pos = relu(left);
    double right_neg_magnitude = relu(-right);
    double outgoing = left_pos > right_neg_magnitude ? left_pos : right_neg_magnitude;
    return (0.5 * outgoing) * outgoing;
}

/* Time just the compiled rollout. Steps and trusted prefix are externally
   chosen. The harness measures 0, N/2 and N, all divisible by LANES. */
int64_t rollout(const double *initial, int neural_faces, int steps, double *result)
{
    struct timespec start, end;
    double state[N], flux[N];
    if (neural_faces < 0 || neural_faces > N || neural_faces % LANES || steps < 0)
        return -1;
    clock_gettime(CLOCK_MONOTONIC, &start);
    memcpy(state, initial, sizeof(state));
    for (int t = 0; t < steps; t++) {
        for (int i = 0; i < neural_faces; i += LANES)
            neural_eight(state, i, flux);
        for (int i = neural_faces; i < N; i++)
            flux[i] = godunov(state[i], state[(i+1) % N]);
        for (int i = 0; i < N; i++) {
            double value = state[i] - 0.2 * (flux[i] - flux[(i+N-1) % N]);
            state[i] = value < -2.0 ? -2.0 : (value > 2.0 ? 2.0 : value);
        }
    }
    memcpy(result, state, sizeof(state));
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec-start.tv_sec) * INT64_C(1000000000) + (end.tv_nsec-start.tv_nsec);
}
