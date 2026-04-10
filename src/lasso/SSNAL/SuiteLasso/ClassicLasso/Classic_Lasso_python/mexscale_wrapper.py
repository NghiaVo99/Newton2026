# mexscale_wrapper.py

import os
import ctypes
from ctypes import c_double, POINTER

# 1) Locate and load the shared library
if os.name == "nt":
    libname = "mexscale.dll"
else:
    libname = "libmexscale.so"

# Assume the .so/.dll sits in the same directory as this file:
_here = os.path.dirname(__file__)
_libpath = os.path.join(_here, libname)
_mex = ctypes.CDLL(_libpath)

# 2) Tell ctypes about mexscale_c’s argument types and return type
#
#    void mexscale_c(
#        double sigma_in,
#        double normx,
#        double normuxi,
#        double bscale_in,
#        double cscale_in,
#        double *sigma_out,
#        double *bscale2_out,
#        double *cscale2_out,
#        double *sbc_out,
#        double *sboc_out,
#        double *bscale_out,
#        double *cscale_out
#    );
#
_mex.mexscale_c.argtypes = (
    c_double,  # sigma_in
    c_double,  # normx
    c_double,  # normuxi
    c_double,  # bscale_in
    c_double,  # cscale_in
    POINTER(c_double),  # sigma_out
    POINTER(c_double),  # bscale2_out
    POINTER(c_double),  # cscale2_out
    POINTER(c_double),  # sbc_out
    POINTER(c_double),  # sboc_out
    POINTER(c_double),  # bscale_out
    POINTER(c_double)   # cscale_out
)
_mex.mexscale_c.restype = None  # it returns void

# 3) Define a pure‐Python wrapper function that allocates outputs,
#    calls into mexscale_c, then returns Python floats
def mexscale(sigma_in: float,
             normx: float,
             normuxi: float,
             bscale_in: float,
             cscale_in: float):
    """
    Calls the C routine mexscale_c and returns a tuple of 7 floats:
      (sigma_out, bscale2_out, cscale2_out,
       sbc_out, sboc_out, bscale_out, cscale_out)
    """
    # Allocate space for the seven outputs as c_double instances
    out_sigma   = c_double()
    out_bscale2 = c_double()
    out_cscale2 = c_double()
    out_sbc     = c_double()
    out_sboc    = c_double()
    out_bscale  = c_double()
    out_cscale  = c_double()

    # Call the C function
    _mex.mexscale_c(
        sigma_in,
        normx,
        normuxi,
        bscale_in,
        cscale_in,
        ctypes.byref(out_sigma),
        ctypes.byref(out_bscale2),
        ctypes.byref(out_cscale2),
        ctypes.byref(out_sbc),
        ctypes.byref(out_sboc),
        ctypes.byref(out_bscale),
        ctypes.byref(out_cscale),
    )

    # Extract Python floats and return them
    return (
        out_sigma.value,
        out_bscale2.value,
        out_cscale2.value,
        out_sbc.value,
        out_sboc.value,
        out_bscale.value,
        out_cscale.value
    )
