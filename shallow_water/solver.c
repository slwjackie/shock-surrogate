/* Intermediate depth of the homogeneous 1-D shallow-water Riemann problem.
   Compile in IEEE binary64 round-to-nearest with gradual underflow,
   -fno-fast-math -ffp-contract=off. See certificate.h for the sign proof. */
#include <math.h>
#include <stddef.h>
#include "certificate.h"
#include "model.h"

enum sw_mode { SW_MEAN, SW_TWO_SHOCK, SW_INITIAL_NET, SW_BOOTSTRAP_NET,
               SW_UNIFORM_NET, SW_TWO_RAREFACTION, SW_FILTERED_NEWTON };

enum sw_route { SW_UNRESOLVED = -3, SW_DRY_OR_UNRESOLVED = -2,
                SW_BAD_INPUT = -1, SW_ANALYTIC = 0, SW_DIRECT, SW_CORRECTED };

static inline double wave_and_slope(double x, double h, double c, double cx,
                                    double g, double *derivative) {
    if (x <= h) {
        *derivative += g/cx;
        /* Rationalized rarefaction difference near endpoint equality. */
        return 2.0*g*(x-h)/(cx+c);
    }
    double scale = sqrt(0.5*g*(1.0/x+1.0/h));
    *derivative += scale*(1.0-0.5*((x-h)/x)*(h/(x+h)));
    return (x-h)*scale;
}

static inline double newton_data(double x, double hl, double hr, double cl, double cr,
                                 double g, double du, double *derivative) {
    *derivative = 0.0;
    double cx = (x <= hl || x <= hr) ? sqrt(g*x) : 0.0;
    double left = wave_and_slope(x,hl,cl,cx,g,derivative);
    double right = wave_and_slope(x,hr,cr,cx,g,derivative);
    return (left+right)+du;
}

/* Two-shock guess of Toro, with the positive primitive-variable depth
   estimate (Muñoz Moncayo et al., arXiv:2209.12235, eqs. 16 and 19). */
static inline double two_shock_guess(double hl, double hr, double cl, double cr,
                                     double g, double du, double rr) {
    double pv = (hl+hr)*(0.5-0.25*du/(cl+cr));
    if (!(pv>0.0) || !isfinite(pv)) pv = rr;
    double yl = sqrt(0.5*g*(1.0/pv+1.0/hl));
    double yr = sqrt(0.5*g*(1.0/pv+1.0/hr));
    return (hl*yl+hr*yr-du)/(yl+yr);
}

/* stats[0] route, [1] Newton updates, [2] bisections, [3] calls to bracket
   certificate (including failed calls). SW_ANALYTIC uses neither NN nor
   Newton for its initial proposal. All returned depths, including Newton
   fallback, must pass sw_certify. Negative status explicitly has no result. */
int sw_solve(double hl, double ul, double hr, double ur, double g, double tol,
             int mode, double *depth, int *stats) {
    if (!depth || !stats) return SW_BAD_INPUT;
    *depth = NAN;
    stats[0] = SW_BAD_INPUT;
    stats[1] = stats[2] = stats[3] = 0;
    if (!(hl>0 && hr>0 && g>0 && tol>=0)
        || !isfinite(hl) || !isfinite(hr) || !isfinite(g) || !isfinite(tol)
        || !isfinite(ul) || !isfinite(ur)
        || mode<SW_MEAN || mode>SW_FILTERED_NEWTON) return stats[0];

    if (hl==hr && ul==ur) {
        /* Algebraically exact even at zero tolerance. */
        stats[3]++;
        if (sw_certify(hl,ul,hr,ur,g,hl,tol)) {
            *depth = hl;
            stats[0] = SW_ANALYTIC;
            return 0;
        }
    }
    double cl = sqrt(g*hl), cr = sqrt(g*hr);
    double cmax = fmax(cl,cr), hmax = fmax(hl,hr);
    double du = ur-ul;
    double cm = 0.5*(cl+cr)-0.25*du;
    if (!(cm>0.0)) return stats[0] = SW_DRY_OR_UNRESOLVED;
    double w = cm/cmax, r = fmin(cl,cr)/cmax;
    double rr = hmax*w*w;
    if (!(rr>0.0) || !isfinite(rr)) return stats[0] = SW_UNRESOLVED;

    /* The exact two-rarefaction formula also upper bounds a wet shock
       solution. The guide bracket is computed in ordinary arithmetic and
       never used by itself to certify an output. */
    int analytic = (w<=r);
    double x;
    if (analytic || mode==SW_TWO_RAREFACTION || mode==SW_FILTERED_NEWTON) {
        x = rr;
    } else if (mode==SW_MEAN) {
        x = 0.5*(hl+hr);
    } else if (mode==SW_TWO_SHOCK) {
        x = two_shock_guess(hl,hr,cl,cr,g,du,rr);
    } else {
        int model = mode==SW_INITIAL_NET ? 0 : (mode==SW_UNIFORM_NET ? 1 : 2);
        x = sw_model_guess(model,hmax,r,w);
    }

    /* Bad neural guesses fall back to a positive Newton seed, never counted
       as a directly accepted proposal. */
    int used_backup = !(x>0.0) || !isfinite(x);
    if (used_backup) x = rr;

    double lo = 0.0;
    double hi = fmax(x,fmax(hmax,rr));
    for (int iter=0; iter<128; ++iter) {
        int check_first = iter==0 && (analytic || mode!=SW_FILTERED_NEWTON);
        if (check_first) {
            stats[3]++;
            if (sw_certify(hl,ul,hr,ur,g,x,tol)) {
                stats[0] = analytic ? SW_ANALYTIC : (used_backup ? SW_CORRECTED : SW_DIRECT);
                *depth = x;
                return 0;
            }
        }

        double derivative;
        double phi = newton_data(x,hl,hr,cl,cr,g,du,&derivative);
        double delta = phi/derivative;
        /* This heuristic saves speculative certificate calls when the next
           Newton displacement is still large. It can only defer acceptance.
           All corrections and all filtered-Newton seeds use this same test. */
        if (!check_first && fabs(delta)<=1.25*tol) {
            stats[3]++;
            if (sw_certify(hl,ul,hr,ur,g,x,tol)) {
                stats[0] = (iter==0 && !used_backup) ? SW_DIRECT : SW_CORRECTED;
                *depth = x;
                return 0;
            }
        }

        if (!isfinite(phi)) return stats[0] = SW_UNRESOLVED;
        if (phi>0) hi = x;
        if (phi<0) lo = x;
        double next = x-delta;
        if (next>lo && next<hi && next!=x && isfinite(next)) {
            stats[1]++;
        } else {
            next = lo+0.5*(hi-lo);
            if (!(next>lo && next<hi) || next==x) break;
            stats[2]++;
        }
        x = next;
    }
    return stats[0] = SW_UNRESOLVED;
}

/* states rows are [h_left,u_left,h_right,u_right]. Returns the count of
   unresolved/invalid rows, or -1 for an invalid batch interface. */
int sw_solve_batch(int n, const double *states, double g, double tol, int mode,
                   double *depths, int *stats) {
    if (n<0 || (n>0 && (!states || !depths || !stats))) return -1;
    int failures = 0;
    for (int i=0; i<n; ++i) {
        const double *s = states+4*(size_t)i;
        if (sw_solve(s[0],s[1],s[2],s[3],g,tol,mode,depths+i,stats+4*(size_t)i)<0)
            ++failures;
    }
    return failures;
}
