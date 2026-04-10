# test_mexscale.py
import os
import ctypes
from ctypes import c_double, POINTER

# 1. Figure out the shared‐library filename
if os.name == "nt":
    # Windows
    libname = "mexscale.dll"
else:
    # Linux / macOS
    libname = "libmexscale.so"

# 2. Load the shared library
libpath = os.path.abspath(libname)
mex = ctypes.CDLL(libpath)

# 3. Tell ctypes about the signature of mexscale_c:
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
mex.mexscale_c.argtypes = (
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
mex.mexscale_c.restype = None   # it returns void

# 4. Allocate space for the 7 outputs (as c_double)
out_sigma   = c_double()
out_bscale2 = c_double()
out_cscale2 = c_double()
out_sbc     = c_double()
out_sboc    = c_double()
out_bscale  = c_double()
out_cscale  = c_double()

# 5. Call the function with your chosen inputs:
sigma_in   = 2.0
normx      = 0.5
normuxi    = 0.3
bscale_in  = 1.0
cscale_in  = 4.0

mex.mexscale_c(
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

# 6. Read back the results:
print("sigma_out:   ", out_sigma.value)
print("bscale2_out: ", out_bscale2.value)
print("cscale2_out: ", out_cscale2.value)
print("sbc_out:     ", out_sbc.value)
print("sboc_out:    ", out_sboc.value)
print("bscale_out:  ", out_bscale.value)
print("cscale_out:  ", out_cscale.value)
