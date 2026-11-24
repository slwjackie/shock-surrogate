/* FP64 periodic Burgers rollout. A common driver times the two compiled
   update kernels without Python dispatch or allocation inside the loop. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "weights.h"

enum { N = 128, H = 8, WIDTH = 16, INPUTS = 10, LANES = 8 };

static inline double positive(double x) { return x > 0.0 ? x : 0.0; }
static inline double face(double left, double right)
{
    double outgoing = positive(left > -right ? left : -right);
    return (0.5 * outgoing) * outgoing;
}

static inline void face_fluxes(const double *restrict state, double *restrict flux)
{
    for (int i = 0; i < N - 1; i++)
        flux[i + 1] = face(state[i], state[i + 1]);
    flux[N] = face(state[N - 1], state[0]);
    flux[0] = flux[N];
}

static inline void godunov_step(const double *restrict state, double *restrict next, double lambda)
{
    double flux[N + 1];
    face_fluxes(state, flux);
    for (int i = 0; i < N; i++)
        next[i] = state[i] - lambda * (flux[i + 1] - flux[i]);
}

static inline void differences(const double *restrict state, double *restrict out)
{
    double flux[N + 1];
    face_fluxes(state, flux);
    for (int i = 0; i < N; i++)
        out[i] = flux[i] - flux[i + 1];
}

static void structured_step(const double *restrict state, double *restrict next,
                            const double *restrict p)
{
    double intermediate[N], d0[N], d1[N], d2[N];
    differences(state, d0);
    for (int i = 0; i < N; i++)
        intermediate[i] = state[i] + p[0] * d0[i];
    differences(intermediate, d1);
    for (int i = 0; i < N; i++)
        intermediate[i] = state[i] + p[1] * d0[i] + p[2] * d1[i];
    differences(intermediate, d2);
    for (int i = 0; i < N; i++)
        next[i] = state[i] + (1.6 - p[3] - p[4]) * d0[i] + p[3] * d1[i] + p[4] * d2[i];
}

static void macro_step(const double *restrict state, double *restrict next)
{
    double pad[N + 7], edge[N + 6], flux[N + 1];
    memcpy(pad, state + N - 3, 3 * sizeof(double));
    memcpy(pad + 3, state, N * sizeof(double));
    memcpy(pad + N + 3, state, 4 * sizeof(double));
    for (int j = 0; j < N + 6; j++)
        edge[j] = pad[j + 1] - pad[j];
    for (int i = 0; i < N; i += LANES) {
        double x[INPUTS][LANES], y[LANES];
        for (int l = 0; l < LANES; l++) {
            double s = 0.0;
            for (int j = 0; j < 7; j++) {
                double a = fabs(edge[i + l + j]);
                s = s > a ? s : a;
            }
            x[0][l] = pad[i + l + 3];
            x[1][l] = pad[i + l + 4];
            x[2][l] = s;
            double denominator = s > 0.0 ? s : 1.0;
            for (int j = 0; j < 7; j++)
                x[j + 3][l] = edge[i + l + j] / denominator;
            y[l] = bias_out;
        }
        for (int k = 0; k < WIDTH; k++) {
            double hidden[LANES];
            for (int l = 0; l < LANES; l++)
                hidden[l] = bias[k];
            for (int j = 0; j < INPUTS; j++)
                for (int l = 0; l < LANES; l++)
                    hidden[l] += x[j][l] * weights[j][k];
            for (int l = 0; l < LANES; l++)
                y[l] += positive(hidden[l]) * weights_out[k];
        }
        for (int l = 0; l < LANES; l++)
            flux[i + l + 1] = face(x[0][l], x[1][l]) + x[2][l] * y[l];
    }
    flux[0] = flux[N];
    for (int i = 0; i < N; i++)
        next[i] = state[i] - 1.6 * (flux[i + 1] - flux[i]);
}

/* Each repetition starts from the same input and returns a complete state.
   The memory clobber ensures the compiler cannot reuse the previous rollout. */
enum Mode { GODUNOV, NEURAL, STRUCTURED, LARGE_GODUNOV, FIXED };

int64_t batch(const double *initial, int mode, int calls, int tail, int repeats, double *result)
{
    struct timespec start, end;
    if (mode < GODUNOV || mode > FIXED || calls < 1 || tail < 0 || tail >= H || repeats < 1)
        return -1;
    int horizon = H * calls + tail;
    if (mode == LARGE_GODUNOV && horizon % 4 != 0)
        return -1;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int rep = 0; rep < repeats; rep++) {
        double workspace[2][N];
        const double *cur = initial;
        if (mode == NEURAL || mode == STRUCTURED || mode == FIXED) {
            int t = 0;
            for (; t < calls; t++) {
                double *next = workspace[t % 2];
                if (mode == NEURAL)
                    macro_step(cur, next);
                else
                    structured_step(cur, next, mode == FIXED ? fixed_weights : structured_weights);
                cur = next;
            }
            for (; t < calls + tail; t++) {
                double *next = workspace[t % 2];
                godunov_step(cur, next, 0.2);
                cur = next;
            }
        } else {
            int steps = mode == LARGE_GODUNOV ? horizon / 4 : horizon;
            double lambda = mode == LARGE_GODUNOV ? 0.8 : 0.2;
            for (int t = 0; t < steps; t++) {
                double *next = workspace[t % 2];
                godunov_step(cur, next, lambda);
                cur = next;
            }
        }
        memcpy(result, cur, N * sizeof(double));
        __asm__ __volatile__("" : : "r"(result) : "memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) * INT64_C(1000000000) + (end.tv_nsec - start.tv_nsec);
}
