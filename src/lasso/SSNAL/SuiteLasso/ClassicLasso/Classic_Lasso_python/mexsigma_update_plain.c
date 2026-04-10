// mexsigma_update_plain.c
#include <math.h>

/* Provide max/min macros just as in the original code */
#ifndef max
#define max(a,b) ((a)>(b)?(a):(b))
#endif
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

/*
 * A “plain‐C” version of mexsigma_update_Classic_Lasso_SSNAL:
 *
 * Inputs:
 *   sigma_in         (double)
 *   sigmamax         (double)
 *   sigmamin         (double)
 *   prim_win_in      (double)
 *   dual_win_in      (double)
 *   iter             (int)
 *   inner_breakyes   (int)
 *
 * Outputs (all via pointers):
 *   *sigma_out
 *   *prim_win_out
 *   *dual_win_out
 */
void mexsigma_update_c(
    double sigma_in,
    double sigmamax,
    double sigmamin,
    double prim_win_in,
    double dual_win_in,
    int    iter,
    int    inner_breakyes,
    /* output pointers: */
    double *sigma_out,
    double *prim_win_out,
    double *dual_win_out
) {
    double sigma    = sigma_in;
    double sigmascale = 5.0;  /* same as original code */
    double prim_win = prim_win_in;
    double dual_win = dual_win_in;
    int sigma_update_iter;

    /* Reproduce the same logic as in your mexFunction: */
    if (iter < 10) {
        sigma_update_iter = 2;
    } else if (iter < 20) {
        sigma_update_iter = 3;
    } else if (iter < 200) {
        sigma_update_iter = 3;
    } else if (iter < 500) {
        sigma_update_iter = 10;
    } else {
        sigma_update_iter = 20;
    }

    if ((iter % sigma_update_iter == 0) && (inner_breakyes < 0)) {
        if (prim_win > max(1.0, 1.2 * dual_win)) {
            prim_win = 0.0;
            sigma = min(sigmamax, sigma * sigmascale);
        } else if (dual_win > max(1.0, 1.2 * prim_win)) {
            dual_win = 0.0;
            sigma = max(sigmamin, sigma / sigmascale);
        }
    }

    if (inner_breakyes >= 0 && iter >= 10) {
        sigma = max(sigmamin, 2.0 * sigma / sigmascale);
    }

    /* Write results back via the output pointers: */
    *sigma_out    = sigma;
    *prim_win_out = prim_win;
    *dual_win_out = dual_win;
}
