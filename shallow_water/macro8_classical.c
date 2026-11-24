/* Build the conventional Newton control and the certified reference into the
   same benchmark library. The prior control includes solver.c, so this unit
   replaces solver.c on the linker command line. */
#include "../reports/shallow_water/engineering_control/newton.c"

static int engineering_depth(double hl, double ul, double hr, double ur,
                             double g, double tol, int mode, double *h, int *stats) {
    const double states[4] = {hl, ul, hr, ur};
    return engineering_batch(1, states, g, tol, mode, h, stats) ? -1 : 0;
}

/* Keep the Riemann sampling, boundary fluxes, update and CFL check identical
   across controls. Only the scalar middle-depth solve is exchanged. */
#define sw_solve engineering_depth
#define sw_rollout sw_engineering_rollout
#include "rollout.c"
#undef sw_rollout
#undef sw_solve
