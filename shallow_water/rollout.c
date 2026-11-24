/* Flat bed, gravity 9.81, first-order finite volumes, transmissive boundaries.
   A certified h_star bounds the scalar interface depth, not the sampled
   momentum/flux or the time-dependent FV trajectory. All flux arithmetic here
   is ordinary binary64, including the wave pattern/speed classification. */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int sw_solve(double hl, double ul, double hr, double ur, double g, double tol,
             int mode, double *depth, int *stats);

static const double gravity = 9.81;

typedef struct { double h, m, u, c, momentum_flux; } Cell;
typedef struct { double mass, momentum; } Flux;

static int read_cell(const double *q, double lambda, Cell *v) {
    v->h = q[0];
    v->m = q[1];
    if (!(v->h > 0) || !isfinite(v->h) || !isfinite(v->m)) return -2;
    v->u = v->m/v->h;
    v->c = sqrt(gravity*v->h);
    if (lambda*(fabs(v->u)+v->c) > 1) return -3;
    v->momentum_flux = v->m*v->u + .5*gravity*v->h*v->h;
    return isfinite(v->momentum_flux) ? 0 : -2;
}

static Flux side_flux(Cell v) {
    return (Flux){v.m, v.momentum_flux};
}

static Flux hu_flux(double h, double u) {
    double m = h*u;
    return (Flux){m, m*u+.5*gravity*h*h};
}

static Flux rusanov(Cell l, Cell r) {
    double a = fmax(fabs(l.u)+l.c, fabs(r.u)+r.c);
    return (Flux){.5*(l.m+r.m-a*(r.h-l.h)),
                  .5*(l.momentum_flux+r.momentum_flux-a*(r.m-l.m))};
}

static Flux hlle(Cell l, Cell r) {
    /* sqrt(h)-weighted Roe speed is equivalently sqrt(g*h)-weighted. */
    double u = (l.c*l.u+r.c*r.u)/(l.c+r.c);
    double c = sqrt(.5*gravity*(l.h+r.h));
    double sl = fmin(l.u-l.c, u-c);
    double sr = fmax(r.u+r.c, u+c);
    if (sl >= 0) return side_flux(l);
    if (sr <= 0) return side_flux(r);
    double inv = 1.0/(sr-sl), product = sl*sr;
    return (Flux){(sr*l.m-sl*r.m+product*(r.h-l.h))*inv,
                  (sr*l.momentum_flux-sl*r.momentum_flux+product*(r.m-l.m))*inv};
}

static double relative_wave(double hstar, double cstar, Cell v) {
    if (hstar <= v.h)
        return 2.0*gravity*(hstar-v.h)/(cstar+v.c);
    return (hstar-v.h)*sqrt(.5*gravity*(1.0/hstar+1.0/v.h));
}

static double edge_speed_bound(double h_upper, Cell v) {
    if (h_upper <= v.h) return v.c;
    return sqrt(.5*gravity*(h_upper/v.h)*(h_upper+v.h));
}

static int godunov(Cell l, Cell r, double tol, int mode, Flux *flux) {
    /* The two-rarefaction depth bounds h_star from above: replacing either
       shock curve by the rarefaction curve decreases the monotone residual.
       The shock-speed magnitude relative to its outside state increases
       with h_star. These tests can select a physical upwind state without
       solving the middle depth. Near zero speed we retain full sampling. */
    double du = r.u-l.u;
    double c_upper = .5*(l.c+r.c) - .25*du;
    if (!(c_upper > 0)) return -4;
    double h_upper = c_upper*c_upper/gravity;
    double guard = 64*DBL_EPSILON*(1+fabs(l.u)+fabs(r.u)+l.c+r.c);
    if (l.u > l.c && l.u-edge_speed_bound(h_upper,l) > guard) {
        *flux = side_flux(l); return 0;
    }
    if (r.u < -r.c && r.u+edge_speed_bound(h_upper,r) < -guard) {
        *flux = side_flux(r); return 0;
    }

    double h;
    int stats[4];
    if (sw_solve(l.h,l.u,r.h,r.u,gravity,tol,mode,&h,stats)<0) return -5;
    double c = sqrt(gravity*h);
    double u = .5*((l.u+r.u) + relative_wave(h,c,r)-relative_wave(h,c,l));
    /* Sample at xi = x/t = 0. A transonic rarefaction contributes its
       interior fan state rather than the middle or exterior state. */
    if (h > l.h) {
        if (l.u-edge_speed_bound(h,l) >= 0) { *flux = side_flux(l); return 0; }
    } else {
        if (l.u-l.c >= 0) { *flux = side_flux(l); return 0; }
        if (u-c >= 0) {
            double ufan = (l.u+2*l.c)/3;
            *flux = hu_flux(ufan*ufan/gravity, ufan); return 0;
        }
    }
    if (h > r.h) {
        if (r.u+edge_speed_bound(h,r) <= 0) { *flux = side_flux(r); return 0; }
    } else {
        if (r.u+r.c <= 0) { *flux = side_flux(r); return 0; }
        if (u+c <= 0) {
            double ufan = (r.u-2*r.c)/3;
            *flux = hu_flux(ufan*ufan/gravity, ufan); return 0;
        }
    }
    *flux = hu_flux(h,u);
    return 0;
}

/* Positive-depth input and output states use [h0,m0,h1,m1,...].
   Success commits the last full step into state, failure leaves it intact.
   -1 invalid arguments, -2 nonphysical/nonfinite result, -3 cell CFL > 1,
   -4 vacuum, -5 unresolved interface depth, -6 insufficient memory. */
int sw_rollout(int n, double *state, double length, double T, int steps,
               double tol, int mode) {
    if (!state || n<2 || (size_t)n>SIZE_MAX/(4*sizeof(double)) ||
        !(length>0) || !isfinite(length) || !(T>0) || !isfinite(T) ||
        steps<=0 || !(tol>=0) || !isfinite(tol) || mode<0 || mode>8) return -1;
    size_t count = 2*(size_t)n;
    double *buffer = malloc(2*count*sizeof(*buffer));
    if (!buffer) return -6;
    double *cur=buffer, *next=buffer+count;
    memcpy(cur,state,count*sizeof(*cur));
    double lambda = (T/steps)/(length/n);
    int status = 0;
    for (int k=0; k<steps; ++k) {
        Cell left, right;
        status = read_cell(cur,lambda,&left);
        if (status) break;
        Flux previous = side_flux(left);
        for (int i=0; i<n; ++i) {
            Flux flux;
            right = left;
            if (i+1 < n) {
                status = read_cell(cur+2*((size_t)i+1),lambda,&right);
                if (status) break;
            }
            if (left.h==right.h && left.m==right.m) flux = side_flux(left);
            else if (mode==7) flux = hlle(left,right);
            else if (mode==8) flux = rusanov(left,right);
            else {
                status = godunov(left,right,tol,mode,&flux);
                if (status) break;
            }
            double h = left.h-lambda*(flux.mass-previous.mass);
            double m = left.m-lambda*(flux.momentum-previous.momentum);
            if (!(h>0) || !isfinite(h) || !isfinite(m)) { status=-2; break; }
            next[2*(size_t)i] = h;
            next[2*(size_t)i+1] = m;
            previous = flux;
            left = right;
        }
        if (status) break;
        double *swap=cur;cur=next;next=swap;
    }
    if (!status) memcpy(state,cur,count*sizeof(*cur));
    free(buffer);
    return status;
}
