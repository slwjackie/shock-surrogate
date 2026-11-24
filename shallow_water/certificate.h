#ifndef SHALLOW_WATER_CERTIFICATE_H
#define SHALLOW_WATER_CERTIFICATE_H

#include <float.h>
#include <math.h>

#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "The shallow-water sign certificate requires IEEE binary64."
#endif
#ifdef __FAST_MATH__
#error "The sign certificate requires strict floating-point arithmetic."
#endif

/* The reference treats the input doubles as exact real numbers. Assume
   round-to-nearest, gradual underflow, correctly rounded elementary binary64
   operations and sqrt, no reassociation. Positive star depth h* solves

       phi(x) = wave(x,hl) + wave(x,hr) + ur - ul = 0,

   where each wave is strictly increasing for x>0, and x=0 is its wet-limit.
   A strict sign is certified; 0 means indeterminate (including exact zero).
   No guarantee on interface flux or trajectory error is implied. */

static inline int sw_cert_depth_scale(double x) {
    return x >= 0x1p-100 && x <= 0x1p100;
}

/* Estimate the residual and its error together. On the fast path the scale
   limits guarantee normal nonzero wave-function intermediates. For a shock
   each reciprocal, sum, product, sqrt and x-h introduce <= one unit roundoff
   (u=2^-53); the sum of the positive reciprocals is well conditioned. Even
   counting sqrt's input errors without attenuation, the wave's absolute error
   is < 12u times its computed magnitude. For a rarefaction, sqrt(g*x)
   and sqrt(g*h) each have relative error < 3u. The subtraction can cancel,
   so the absolute error is bounded using the sum of the positive wave speeds,
   including their factor of two. The last two residual additions and ur-ul
   add at most 6u times the sum of all magnitudes. Accumulating the nonnegative
   scale takes at most four additions (relative error < 5u). 64u covers all
   these bounds; DBL_MIN also covers subnormal rounding in the velocity
   difference, residual or error bound. This filter is a sufficient sign check. */
static inline double sw_cert_wave(double x, double h, double g, double *scale) {
    if (x == h) return 0.0;
    if (x < h) {
        double a = sqrt(g*x), b = sqrt(g*h);
        *scale += 2.0*(a+b);
        return 2.0*(a-b);
    }
    double f = (x-h)*sqrt((0.5*g)*(1.0/x+1.0/h));
    *scale += f;
    return f;
}

/* Tighten ambiguous signs and support poorly scaled states by enclosing
   individual operations. The endpoint expressions are monotone in their
   positive inputs. All bounds expand by one representable number. */
static inline double sw_cert_lo(double x) { return nextafter(x,-INFINITY); }
static inline double sw_cert_hi(double x) { return nextafter(x, INFINITY); }

static inline void sw_cert_wave_bounds(double x, double h, double g,
                                       double *low, double *high) {
    if (x == h) {
        *low = *high = 0.0;
        return;
    }
    if (x < h) {
        double gx = g*x, gh = g*h;
        double ax_lo = sw_cert_lo(sqrt(fmax(0.0,sw_cert_lo(gx))));
        double ax_hi = sw_cert_hi(sqrt(sw_cert_hi(gx)));
        double ah_lo = sw_cert_lo(sqrt(fmax(0.0,sw_cert_lo(gh))));
        double ah_hi = sw_cert_hi(sqrt(sw_cert_hi(gh)));
        *low = sw_cert_lo(2.0*sw_cert_lo(ax_lo-ah_hi));
        *high = sw_cert_hi(2.0*sw_cert_hi(ax_hi-ah_lo));
        return;
    }
    double rx = 1.0/x, rh = 1.0/h;
    double lo_sum = fmax(0.0,sw_cert_lo(sw_cert_lo(rx)+sw_cert_lo(rh)));
    double hi_sum = sw_cert_hi(sw_cert_hi(rx)+sw_cert_hi(rh));
    double lo_g = fmax(0.0,sw_cert_lo(0.5*g));
    double hi_g = sw_cert_hi(0.5*g);
    double lo_root = sw_cert_lo(sqrt(fmax(0.0,sw_cert_lo(lo_g*lo_sum))));
    double hi_root = sw_cert_hi(sqrt(sw_cert_hi(hi_g*hi_sum)));
    *low = sw_cert_lo(fmax(0.0,sw_cert_lo(x-h))*fmax(0.0,lo_root));
    *high = sw_cert_hi(sw_cert_hi(x-h)*hi_root);
}

static inline int sw_sign_at(double hl, double ul, double hr, double ur,
                             double g, double x) {
    if (!(hl>0 && hr>0 && g>0 && x>=0) || !isfinite(hl) || !isfinite(hr)
        || !isfinite(g) || !isfinite(x) || !isfinite(ul) || !isfinite(ur)) return 0;
    if ((sw_cert_depth_scale(x) || x==0) && sw_cert_depth_scale(hl)
        && sw_cert_depth_scale(hr) && sw_cert_depth_scale(g)
        && fabs(ul)<=0x1p100 && fabs(ur)<=0x1p100) {
        double scale = fabs(ul)+fabs(ur);
        double left = sw_cert_wave(x,hl,g,&scale);
        double right = sw_cert_wave(x,hr,g,&scale);
        double r = (left+right)+(ur-ul);
        double error = 0x1p-47*scale+DBL_MIN; /* 64 unit roundoffs */
        if (r > error) return 1;
        if (r < -error) return -1;
    }
    double ll, lu, rl, ru;
    sw_cert_wave_bounds(x,hl,g,&ll,&lu);
    sw_cert_wave_bounds(x,hr,g,&rl,&ru);
    double du = ur-ul;
    double lo = sw_cert_lo(sw_cert_lo(ll+rl)+sw_cert_lo(du));
    double hi = sw_cert_hi(sw_cert_hi(lu+ru)+sw_cert_hi(du));
    if (isnan(lo) || isnan(hi)) return 0;
    return lo>0 ? 1 : (hi<0 ? -1 : 0);
}

static inline int sw_certify(double hl, double ul, double hr, double ur,
                             double g, double candidate, double tol) {
    if (!(candidate>0 && tol>=0 && hl>0 && hr>0 && g>0) || !isfinite(candidate)
        || !isfinite(tol) || !isfinite(hl) || !isfinite(hr) || !isfinite(g)
        || !isfinite(ul) || !isfinite(ur)) return 0;
    if (hl==hr && ul==ur && candidate==hl) return 1;
    /* Inward rounding prevents an endpoint one ulp beyond the proposed
       tolerance being used as evidence. At x=0 the sign must still be
       strictly negative, excluding dry-middle/vacuum states. */
    double low = candidate<=tol ? 0.0 : sw_cert_hi(candidate-tol);
    double high = sw_cert_lo(candidate+tol);
    return low<high && sw_sign_at(hl,ul,hr,ur,g,low)<0
                    && sw_sign_at(hl,ul,hr,ur,g,high)>0;
}

#endif
