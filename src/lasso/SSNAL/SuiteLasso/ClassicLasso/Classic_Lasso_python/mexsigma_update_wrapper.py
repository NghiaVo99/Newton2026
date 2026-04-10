# mexsigma_update_wrapper.py

import os
import ctypes
from ctypes import c_double, c_int, POINTER

# 1) Determine the shared‐library name based on OS
if os.name == "nt":
    libname = "mexsigma_update.dll"
else:
    libname = "libmexsigma_update.so"

# 2) Construct the full path (assumes the .so/.dll is in the same directory)
_here = os.path.dirname(__file__)
_libpath = os.path.join(_here, libname)

# 3) Load the shared library
_mexlib = ctypes.CDLL(_libpath)

# 4) Declare mexsigma_update_c’s signature:
#
#    void mexsigma_update_c(
#        double sigma_in,
#        double sigmamax,
#        double sigmamin,
#        double prim_win_in,
#        double dual_win_in,
#        int    iter,
#        int    inner_breakyes,
#        double *sigma_out,
#        double *prim_win_out,
#        double *dual_win_out
#    );
#
_mexlib.mexsigma_update_c.argtypes = (
    c_double,   # sigma_in
    c_double,   # sigmamax
    c_double,   # sigmamin
    c_double,   # prim_win_in
    c_double,   # dual_win_in
    c_int,      # iter
    c_int,      # inner_breakyes
    POINTER(c_double),  # sigma_out
    POINTER(c_double),  # prim_win_out
    POINTER(c_double)   # dual_win_out
)
_mexlib.mexsigma_update_c.restype = None  # void

# 5) Python‐friendly wrapper:
def mexsigma_update_Classic_Lasso_SSNAL(sigma_in: float,
                    sigmamax: float,
                    sigmamin: float,
                    prim_win: float,
                    dual_win: float,
                    iter: int,
                    inner_breakyes: int):
    """
    Calls the C routine mexsigma_update_c and returns a tuple of three floats:
      (sigma_out, prim_win_out, dual_win_out)
    """
    # Allocate space for the three output doubles
    out_sigma    = c_double()
    out_prim_win = c_double()
    out_dual_win = c_double()

    # Call the C function
    _mexlib.mexsigma_update_c(
        sigma_in,
        sigmamax,
        sigmamin,
        prim_win,
        dual_win,
        iter,
        inner_breakyes,
        ctypes.byref(out_sigma),
        ctypes.byref(out_prim_win),
        ctypes.byref(out_dual_win),
    )

    # Return Python floats
    return out_sigma.value, out_prim_win.value, out_dual_win.value
