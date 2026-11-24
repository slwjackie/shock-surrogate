/* Eight-step blocks for the 256-cell dam benchmark. Across the original gate,
   both adjacent cells use the Godunov time-integrated transfer divided by dx. */
#include <math.h>
#include <string.h>

int sw_rollout_with_transfer(int, double *, double, double, int, double, int, int, double *);
int sw_macro8_predict_with_transfer(int, const double *, double *, int, int, double *);

enum { CELLS = 256, STEPS = 8, REPLAY_MODEL = 1 };
enum { GODUNOV = 0, NEURAL = 1, SPATIAL = 2 };

static int godunov(double *q, int n, int face, double *transfer) {
    /* Preserve the full mesh's dx and dt even on a cropped stencil. */
    return sw_rollout_with_transfer(n,q,(double)n/CELLS,STEPS*.07/CELLS,
                                    STEPS,1e-8,6,face,transfer);
}

static int spatial_step(const double *q, double *next, int gate, int deep_left) {
    /* Both methods need at most eight input cells beyond the gate. */
    int neural_start = deep_left ? 0 : gate-STEPS;
    int neural_end = deep_left ? gate+STEPS : CELLS;
    int god_start = deep_left ? gate-STEPS : 0;
    int god_end = deep_left ? CELLS : gate+STEPS;
    double neural_patch[2*CELLS], god_patch[2*CELLS];
    double neural_transfer[2], god_transfer[2];
    int code = sw_macro8_predict_with_transfer(neural_end-neural_start,q+2*neural_start,
                    neural_patch,REPLAY_MODEL,gate-neural_start,neural_transfer);
    if (code) return code;
    memcpy(god_patch,q+2*god_start,2*(god_end-god_start)*sizeof(double));
    code = godunov(god_patch,god_end-god_start,gate-god_start,god_transfer);
    if (code) return code;

    const double *left = deep_left ? neural_patch : god_patch;
    const double *right = deep_left ? god_patch : neural_patch;
    memcpy(next,left,2*gate*sizeof(double));
    memcpy(next+2*gate,right+2*STEPS,2*(CELLS-gate)*sizeof(double));
    int neural_cell = deep_left ? gate-1 : gate;
    for (int v=0; v<2; ++v) {
        double change = god_transfer[v]-neural_transfer[v];
        next[2*neural_cell+v] += deep_left ? -change : change;
    }
    return 0;
}

/* The input is immutable. Failure identifies the first unfinished block. */
int sw_run_dam(const double *initial, double *result, int method, int gate,
               int deep_left, int blocks, int *failed_block) {
    if (!initial || !result || !failed_block || initial==result ||
        method<GODUNOV || method>SPATIAL || gate<STEPS || gate>CELLS-STEPS ||
        blocks<1 || blocks>8) return -1;
    double q[2*CELLS], next[2*CELLS];
    memcpy(q,initial,sizeof(q));
    for (int block=1; block<=blocks; ++block) {
        *failed_block=block;
        int code;
        if (method==GODUNOV) {
            memcpy(next,q,sizeof(q));
            code=godunov(next,CELLS,0,NULL);
        } else if (method==NEURAL) {
            code=sw_macro8_predict_with_transfer(CELLS,q,next,REPLAY_MODEL,0,NULL);
        } else {
            code=spatial_step(q,next,gate,deep_left);
        }
        if (code) return code;
        for (int i=0; i<CELLS; ++i) {
            if (!(next[2*i]>0.) || !isfinite(next[2*i]) || !isfinite(next[2*i+1]))
                return -2;
        }
        memcpy(q,next,sizeof(q));
    }
    *failed_block=0;
    memcpy(result,q,sizeof(q));
    return 0;
}
