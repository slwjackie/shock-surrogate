/* Empirical postcheck for the fixed 8-step, lambda=.07 operator.
   All inputs are interleaved depth h and discharge m=hu. The score compares
   an approximate time-integrated balance with the predicted change over short
   spatial intervals. Calibration is empirical; the score is not an error bound. */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int sw_macro8_predict(int n, const double *state, double *out, int model);
int sw_rollout(int n, double *state, double length, double duration, int steps,
               double tol, int mode);

enum { micro_steps = 8, patch_width = 2 };
static const double block_lambda = 8.0*0.07;
static const double gravity = 9.81;
static const double depth_target = 0.01;
static const double discharge_target = 0.04;

typedef struct { double h, m; } Pair;

static Pair cell_flux(const double *cell) {
    double h = cell[0], m = cell[1];
    return (Pair){m, m*(m/h) + 0.5*gravity*h*h};
}

static Pair mean_endpoint_flux(const double *begin, const double *end) {
    Pair a = cell_flux(begin), b = cell_flux(end);
    return (Pair){0.5*(a.h+b.h), 0.5*(a.m+b.m)};
}

/* Eight equal cells at either edge ensure a fixed outward face flux at every
   microstep of the reference stencil. Else the flux integral would require
   boundary history, and this guard abstains. */
static int constant_edges(int n, const double *q) {
    for (int j=1; j<micro_steps; ++j) {
        size_t left = 2*(size_t)j, right = 2*(size_t)(n-1-j);
        size_t last = 2*(size_t)(n-1);
        if (q[left] != q[0] || q[left+1] != q[1] ||
            q[right] != q[last] || q[right+1] != q[last+1]) return 0;
    }
    return 1;
}

/* Returns 1 on acceptance and 0 on rejection. A hard failure stores infinite
   risk; a finite score may be read by passing +infinity as max_risk. */
int sw_macro8_check(int n, const double *q, const double *pred,
                    double max_risk, double *risk) {
    if (!risk) return 0;
    *risk = INFINITY;
    if (!q || !pred || n < 2*micro_steps || isnan(max_risk) ||
        (size_t)n > SIZE_MAX/(2*sizeof(double)) || !constant_edges(n,q)) return 0;

    Pair total={0,0}, absolute_scale={0,0}, local={0,0}, defects={0,0};
    Pair previous_flux={0,0};
    for (int i=0; i<n; ++i) {
        size_t j=2*(size_t)i;
        if (!(q[j]>0) || !(pred[j]>0) || !isfinite(q[j]) || !isfinite(q[j+1]) ||
            !isfinite(pred[j]) || !isfinite(pred[j+1])) return 0;
        Pair dq={pred[j]-q[j],pred[j+1]-q[j+1]};
        total.h+=dq.h;
        total.m+=dq.m;
        absolute_scale.h+=fabs(q[j])+fabs(pred[j]);
        absolute_scale.m+=fabs(q[j+1])+fabs(pred[j+1]);
        if (i==0 || i%patch_width==0 || i==n-1) {
            /* Cut a control volume at a cell centre, allocating half that
               cell's change to each adjacent volume. Approximate the time
               integral of the physical flux by the trapezoidal rule. */
            Pair flux=mean_endpoint_flux(q+j,pred+j);
            if (i>0) {
                defects.h+=fabs(local.h+.5*dq.h+block_lambda*(flux.h-previous_flux.h));
                defects.m+=fabs(local.m+.5*dq.m+block_lambda*(flux.m-previous_flux.m));
            }
            previous_flux=flux;
            local=(Pair){.5*dq.h,.5*dq.m};
        } else {
            local.h+=dq.h;
            local.m+=dq.m;
        }
    }
    Pair left=cell_flux(q), right=cell_flux(q+2*(size_t)(n-1));
    Pair net={block_lambda*(left.h-right.h),block_lambda*(left.m-right.m)};
    /* Scaled floating-point tolerance for this diagnostic identity. */
    double roundoff=32*DBL_EPSILON*(n+1.);
    if (!isfinite(total.h) || !isfinite(total.m) ||
        !isfinite(absolute_scale.h) || !isfinite(absolute_scale.m) ||
        !isfinite(net.h) || !isfinite(net.m) ||
        !isfinite(defects.h) || !isfinite(defects.m) ||
        fabs(total.h-net.h)>roundoff*(1+absolute_scale.h+fabs(net.h)) ||
        fabs(total.m-net.m)>roundoff*(1+absolute_scale.m+fabs(net.m))) return 0;
    double score=fmax(defects.h/(n*depth_target),defects.m/(n*discharge_target));
    if (!isfinite(score)) return 0;
    *risk=score;
    return score<=max_risk;
}

/* Public route: one predict/postcheck attempt, then a full eight-step rollout
   from the original input if required. Only a successful result is copied to
   out, including when caller storage aliases q. 1=NN, 0=fallback, <0=failure. */
int sw_macro8_routed(int n, const double *q, double *out, int model,
                     double max_risk, double *risk) {
    if (!risk) return -1;
    *risk=INFINITY;
    if (!q || !out || n<2 || (size_t)n>SIZE_MAX/(2*sizeof(double)) ||
        model<0 || model>2 || isnan(max_risk)) return -1;
    size_t bytes=2*(size_t)n*sizeof(double);
    double *trial=malloc(bytes);
    if (!trial) return -6;
    int route=0;
    if (sw_macro8_predict(n,q,trial,model)==0 &&
        sw_macro8_check(n,q,trial,max_risk,risk)) route=1;
    else {
        memcpy(trial,q,bytes);
        route=sw_rollout(n,trial,1.0,block_lambda/n,micro_steps,1e-8,6);
    }
    if (route>=0) memcpy(out,trial,bytes);
    free(trial);
    return route;
}
