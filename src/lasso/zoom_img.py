# zoom_panel_native.py
import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib.patches import Rectangle

plt.rcParams['image.interpolation'] = 'nearest'   # never smooth pixels

# -----------------------------
# 1) Load stack and mean image
# -----------------------------
#files = sorted(glob.glob("reconstructed/*.tif"))
files = sorted(glob.glob("sequence/*.tif"))
if not files:
    raise FileNotFoundError("No .tif files in reconstructed/*.tif")
imgs = [tiff.imread(f).astype(np.float32) for f in files]  # assumes 2D, same size
mean_img = np.mean(np.stack(imgs, axis=0), axis=0)
H, W = mean_img.shape

# -----------------------------
# 2) Choose ROI
#    Option A: set numbers below
#    Option B: set USE_INTERACTIVE = True to drag a box
# -----------------------------
USE_INTERACTIVE = False

# --- Option A (edit these four integers) ---
y1, y2 = 140, 240   # rows: top, bottom
x1, x2 =  30, 130   # cols: left, right

# --- Option B (interactive picker) ---
if USE_INTERACTIVE:
    from matplotlib.widgets import RectangleSelector
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(mean_img, cmap="gist_heat", origin="upper")
    ax.set_title("Drag to choose ROI, then close this window")
    ax.set_axis_off()
    _roi = {"coords": None}

    def onselect(eclick, erelease):
        x1i, y1i = eclick.xdata,  eclick.ydata
        x2i, y2i = erelease.xdata, erelease.ydata
        if None in (x1i, y1i, x2i, y2i):
            return
        xs = sorted([int(round(x1i)), int(round(x2i))])
        ys = sorted([int(round(y1i)), int(round(y2i))])
        _roi["coords"] = (max(0, ys[0]), min(H, ys[1]), max(0, xs[0]), min(W, xs[1]))
        print("ROI:", _roi["coords"])

    rs = RectangleSelector(ax, onselect, useblit=True, button=[1],
                           minspanx=3, minspany=3, spancoords='pixels', interactive=True)
    plt.show()
    if _roi["coords"]:
        y1, y2, x1, x2 = _roi["coords"]

# clip & validate
y1, y2 = max(0, y1), min(H, y2)
x1, x2 = max(0, x1), min(W, x2)
assert y2 > y1 and x2 > x1, "Invalid ROI"

# -----------------------------
# 3) Utility: save with native pixels
# -----------------------------
def save_native_png(path, img, cmap="inferno"):
    """Save exactly at array's native pixel size (no extra pixels)."""
    h, w = img.shape[:2]
    dpi = 100  # any value; figsize is matched to force native size
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.imshow(img, cmap=cmap, origin="upper", interpolation="nearest")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


MAG = 8   # try 4, 6, 8, ...

roi = mean_img[y1:y2, x1:x2]
save_native_png("mean_roi_native.png", roi, cmap="gist_heat")

# ---- FULL WITH BOX ----
fig1, ax1 = plt.subplots(figsize=(8, 10), dpi=120)
ax1.imshow(mean_img, cmap="gist_heat", origin="upper", interpolation="nearest")
ax1.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                        fill=False, edgecolor="green", linewidth=2))
ax1.set_axis_off()
fig1.savefig("full_with_box.png", dpi=120, bbox_inches="tight")
plt.close(fig1)

# ---- ROI MAGNIFIED (pixel-true nearest) ----
roi = mean_img[y1:y2, x1:x2]
roi_h, roi_w = roi.shape
MAG = 6               # make this larger to magnify more (no smoothing)
dpi = 120
fig_w = roi_w * MAG / dpi
fig_h = roi_h * MAG / dpi

fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
ax2.imshow(roi, cmap="gist_heat", origin="upper", interpolation="nearest")
ax2.set_axis_off()
fig2.savefig("roi_mag.png", dpi=dpi, bbox_inches="tight")




plt.show()

