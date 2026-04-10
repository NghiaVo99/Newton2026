import pathlib, numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from math import ceil

FOLDER = pathlib.Path("sequence")
#FOLDER = pathlib.Path("reconstructed")
SHOW_N = 3          # how many images to display
COLS = 3            # grid columns

# collect image files (you can add more extensions if needed)
files = sorted([
    *FOLDER.glob("*.tif"), *FOLDER.glob("*.tiff"),
    *FOLDER.glob("*.png"), *FOLDER.glob("*.jpg"), *FOLDER.glob("*.jpeg")
])

if not files:
    raise FileNotFoundError(f"No images found in {FOLDER.resolve()}")

def load_for_display(path):
    # If a TIFF has multiple pages, index=0 loads the first page
    img = iio.imread(path, index=0)
    img = np.asarray(img)
    # grayscale if needed
    if img.ndim == 3 and img.shape[-1] in (3,4):  # RGB/RGBA → luminance
        img = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    # normalize to [0,1] for display
    img = img.astype(np.float64)
    mn, mx = np.min(img), np.max(img)
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return img

imgs = [load_for_display(p) for p in files[:SHOW_N]]

rows = ceil(len(imgs)/COLS)
plt.figure(figsize=(COLS*3.2, rows*3.2))
for i, (p, im) in enumerate(zip(files, imgs), 1):
    ax = plt.subplot(rows, COLS, i)
    ax.imshow(im, cmap="gist_heat", interpolation="nearest")
    ax.set_title(p.name, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
