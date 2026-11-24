/* Independent profiles in a single native call. All methods take immutable
   named input and write an output profile, including on Godunov fallback. */
#include <math.h>
#include <stddef.h>
#include <string.h>

int sw_rollout(int, double *, double, double, int, double, int);
int sw_engineering_rollout(int, double *, double, double, int, double, int);
int sw_macro8_predict(int, const double *, double *, int);
int sw_macro8_routed(int, const double *, double *, int, double, double *);

enum method {
    TEACHER8, GODUNOV8, GODUNOV_CFL, ENGINEERING8, ENGINEERING_CFL,
    HLLE8, HLLE_CFL, RUSANOV8, RUSANOV_CFL, INITIAL, UNIFORM, HARD,
    ROUTED_INITIAL, ROUTED_UNIFORM, ROUTED_HARD
};

/* Both inputs represent conserved variables (h, hu) as in rollout.c. The
   initial wave-speed estimates can increase during a block; the production
   kernels check their CFL on every step. If a larger step fails, retry with
   eight original steps from the untouched initial profile and time both. */
int sw_macro8_classical(int n, const double *q, double *out, int method, int *stats) {
    if (n < 2 || !q || !out || !stats || method < TEACHER8 || method > RUSANOV_CFL)
        return -1;
    const int reduced = method == GODUNOV_CFL || method == ENGINEERING_CFL ||
                        method == HLLE_CFL || method == RUSANOV_CFL;
    int steps = 8;
    if (reduced) {
        double maxspeed = 0;
        for (int i = 0; i < n; ++i) {
            double h = q[2*(size_t)i], m = q[2*(size_t)i+1];
            if (!(h > 0) || !isfinite(h) || !isfinite(m)) return -2;
            maxspeed = fmax(maxspeed, fabs(m/h)+sqrt(9.81*h));
        }
        steps = (int)fmin(8.0, fmax(1.0, ceil((8.0*.07)*maxspeed/.90)));
    }
    int mode = 5;
    if (method == TEACHER8) mode = 6;
    if (method == HLLE8 || method == HLLE_CFL) mode = 7;
    if (method == RUSANOV8 || method == RUSANOV_CFL) mode = 8;
    const double tol = method == TEACHER8 ? 1e-8 : .01;
    const double duration = (8.0*.07)/n;
    const int engineering = method == ENGINEERING8 || method == ENGINEERING_CFL;
    int (*advance)(int, double *, double, double, int, double, int) =
        engineering ? sw_engineering_rollout : sw_rollout;
    memcpy(out, q, 2*(size_t)n*sizeof(double));
    int status = advance(n, out, 1., duration, steps, tol, mode);
    stats[0] = -1;
    stats[1] = steps;
    stats[2] = 0;
    if ((status == -3 || status == -2) && steps < 8) {
        stats[1] = 8;
        stats[2] = 1;
        memcpy(out, q, 2*(size_t)n*sizeof(double));
        status = advance(n, out, 1., duration, 8, tol, mode);
    }
    return status;
}

/* stats is [routes, steps, retries] per case. For routed nets route is 1 for
   learned, 0 for fallback. For a bare learned or classical method it is -1. */
int sw_macro8_batch(int batch, int n, const double *q, double *out, int method,
                    double threshold, int *stats) {
    if (n < 2 || batch < 1 || !q || !out || !stats || method < TEACHER8 ||
        method > ROUTED_HARD) return -1;
    for (int j = 0; j < batch; ++j) {
        size_t k = 2*(size_t)j*(size_t)n;
        int *record = stats+3*(size_t)j;
        record[0] = -1;
        record[1] = record[2] = 0;
        int status;
        if (method <= RUSANOV_CFL) {
            status = sw_macro8_classical(n, q+k, out+k, method, record);
        } else if (method < ROUTED_INITIAL) {
            status = sw_macro8_predict(n, q+k, out+k, method-INITIAL);
        } else {
            double risk;
            status = sw_macro8_routed(n, q+k, out+k, method-ROUTED_INITIAL, threshold, &risk);
            if (status >= 0) {
                record[0] = status;
                record[1] = status == 0 ? 8 : 0;
                status = 0;
            }
        }
        if (status) return status;
    }
    return 0;
}
