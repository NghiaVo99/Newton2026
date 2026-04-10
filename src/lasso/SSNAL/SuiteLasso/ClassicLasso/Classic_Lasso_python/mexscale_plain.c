// mexscale_plain.c
#include <math.h>

/*
 * A “plain‐C” version of the same logic in mexscale.c,
 * but without any MATLAB/MEX dependencies.
 *
 * Original inputs (5):   sigma_in, normx, normuxi, bscale_in, cscale_in
 * Original outputs (7):  sigma_out, bscale2_out, cscale2_out,
 *                       sbc_out, sboc_out, bscale_out, cscale_out
 */
void mexscale_c(
    double sigma_in,
    double normx,
    double normuxi,
    double bscale_in,
    double cscale_in,
    /* pointers for outputs: */
    double *sigma_out,
    double *bscale2_out,
    double *cscale2_out,
    double *sbc_out,
    double *sboc_out,
    double *bscale_out,
    double *cscale_out
) {
    double bscale2, cscale2, cst = 1.0;
    double sbc, sboc;
    double sigma = sigma_in;
    double bscale = bscale_in;
    double cscale = cscale_in;

    /* Exactly the same math you had in mexFunction: */
    if (normx < 1e-7) {
        normx   = 1.0;
        normuxi = 1.0;
    }
    bscale2 = normx * cst;
    cscale2 = normuxi * cst;
    sbc     = sqrt(bscale2 * cscale2);
    sboc    = sqrt(bscale2 / cscale2);
    sigma   = sigma * (cscale2 / bscale2);
    cscale  = cscale * cscale2;
    bscale  = bscale * bscale2;

    /* Store results back via the pointers: */
    *sigma_out    = sigma;
    *bscale2_out  = bscale2;
    *cscale2_out  = cscale2;
    *sbc_out      = sbc;
    *sboc_out     = sboc;
    *bscale_out   = bscale;
    *cscale_out   = cscale;
}
