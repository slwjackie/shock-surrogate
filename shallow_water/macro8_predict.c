/* One pass from a whole (h,hu) field to the endpoint of eight steps with
   dt/dx = 0.07. Input/output arrays are disjoint, interleaved binary64. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "macro8_weights.h"

enum { R = 7, TAPS = 15, WIDTH = 8, CHANNELS = 18 };

int sw_macro8_predict_with_transfer(int n, const double *q, double *out, int model,
                                    int face, double *transfer) {
    if (transfer && (face < 1 || face >= n)) return -1;
    if (!q || !out || q == out || n < 2 || model < 0 || model > 2 ||
        (size_t)n > SIZE_MAX/(2*sizeof(double))-2*R-1) return -1;
    const double c = sqrt(9.81);
    const double courant = (8.*.07)*c;
    /* Ghost faces make every spatial convolution the same length. At the
       two physical boundary faces the flux is fixed again before update. */
    size_t faces = (size_t)n+2*R+1;
    double (*flux)[2] = calloc(faces, sizeof(*flux));
    if (!flux) return -2;
    double last_h = 0., last_m = 0., last_u = 0., last_momentum_flux = 0.;
    double left0 = 0., left1 = 0.;
    int status = 0;
    for (int i=0; i<n; ++i) {
        size_t k = 2*(size_t)i;
        double h = q[k], m = q[k+1]/c;
        if (!(h>0.) || !isfinite(h) || !isfinite(m)) { status=-3; break; }
        double u = m/h, momentum_flux = m*u + .5*h*h;
        if (!isfinite(momentum_flux)) { status=-3; break; }
        if (i == 0) {
            left0 = m;
            left1 = momentum_flux;
        } else {
            flux[i+R][0] += .5*(last_m+m);
            flux[i+R][1] += .5*(last_momentum_flux+momentum_flux);
            double dh = h-last_h, dm = m-last_m;
            if (dh != 0. || dm != 0.) {
                double du = u-last_u;
                double x[] = {sqrt(.5*(last_h+h))-1., .5*(last_u+u), dh, du, fabs(dh), fabs(du)};
                double embed[CHANNELS];
                embed[0] = dh;
                embed[1] = dm;
                for (int a=0; a<WIDTH; ++a) {
                    double z = sw8_b[model][a];
                    for (int f=0; f<6; ++f) z += x[f]*sw8_w[model][f][a];
                    double hidden = fmax(z,0.);
                    embed[2+a] = dh*hidden;
                    embed[2+WIDTH+a] = dm*hidden;
                }
                for (int a=0; a<CHANNELS; ++a) {
                    for (int t=0; t<TAPS; ++t) {
                        flux[i+t][0] += embed[a]*sw8_k[model][a][TAPS-1-t][0];
                        flux[i+t][1] += embed[a]*sw8_k[model][a][TAPS-1-t][1];
                    }
                }
            }
        }
        last_h = h;
        last_m = m;
        last_u = u;
        last_momentum_flux = momentum_flux;
    }
    if (!status) {
        flux[R][0] = left0;
        flux[R][1] = left1;
        flux[(size_t)n+R][0] = last_m;
        flux[(size_t)n+R][1] = last_momentum_flux;
        if (transfer) {
            transfer[0] = courant*flux[face+R][0];
            transfer[1] = (courant*flux[face+R][1])*c;
        }
        for (int i=0; i<n; ++i) {
            size_t k = 2*(size_t)i, left = (size_t)i+R;
            out[k] = q[k] + courant*(flux[left][0]-flux[left+1][0]);
            out[k+1] = q[k+1] + (courant*(flux[left][1]-flux[left+1][1]))*c;
        }
    }
    free(flux);
    return status;
}

int sw_macro8_predict(int n, const double *q, double *out, int model) {
    return sw_macro8_predict_with_transfer(n,q,out,model,0,NULL);
}
