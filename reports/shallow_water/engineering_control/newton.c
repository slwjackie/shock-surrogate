/* Standalone benchmark translation unit; reuse the same classical
   wave evaluations, seeds and native ABI as the production solver. */
#include "../../../shallow_water/solver.c"

int engineering_batch(int n, const double *states, double g, double tol, int mode,
                      double *depth, int *stats) {
    int failures = 0;
    for (int i=0; i<n; ++i) {
        const double *s = states+4*(size_t)i;
        int *cost = stats+4*(size_t)i;
        double hl=s[0], ul=s[1], hr=s[2], ur=s[3];
        double cl=sqrt(g*hl), cr=sqrt(g*hr), du=ur-ul;
        double cm=0.5*(cl+cr)-0.25*du;
        cost[0]=cost[1]=cost[2]=cost[3]=0;
        depth[i]=NAN;
        if (!isfinite(du) || !isfinite(cm) || !(hl>0 && hr>0 && cm>0 && g>0 && tol>0)) {
            ++failures;
            continue;
        }
        double rr=cm*cm/g;
        if (cm<=fmin(cl,cr)) {
            depth[i]=rr;
            continue;
        }
        double x=(mode==SW_TWO_SHOCK) ? two_shock_guess(hl,hr,cl,cr,g,du,rr) : rr;
        if (!(x>0.0) || !isfinite(x)) x=rr;
        double lo=0.0, hi=fmax(x,fmax(rr,fmax(hl,hr)));
        for (int k=0; k<128; ++k) {
            double derivative;
            double phi=newton_data(x,hl,hr,cl,cr,g,du,&derivative);
            double delta=phi/derivative;
            if (phi>0.0) hi=x;
            if (phi<0.0) lo=x;
            double next=x-delta;
            if (!isfinite(next) || next<=lo || next>=hi) {
                if (next!=x) {
                    next=lo+0.5*(hi-lo);
                    ++cost[2];
                }
            } else {
                ++cost[1];
            }
            if (isfinite(next) && next>0 && fabs(delta)<=0.1*tol) {
                depth[i]=next;
                break;
            }
            x=next;
        }
        if (!isfinite(depth[i])) ++failures;
    }
    return failures;
}
