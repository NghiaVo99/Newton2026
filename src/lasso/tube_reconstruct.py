import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff  
from matplotlib import cm, colors
from pathlib import Path
from benchmarks import paper_settings

# 1) Point to your reconstructed folder (adjust the path/pattern if needed)
SCRIPT_DIR = Path(__file__).resolve().parent
files = sorted(glob.glob(str(SCRIPT_DIR / "reconstructed/*.tif")))
files1 = sorted(glob.glob(str(SCRIPT_DIR / "sequence/*.tif")))
expected_frames = paper_settings.FIGURES_8_9_POISSON["n_frames"]
if len(files) != expected_frames or len(files1) != expected_frames:
    raise ValueError(
        f"Figure 9 requires {expected_frames} raw and reconstructed frames; "
        f"found {len(files1)} raw and {len(files)} reconstructed."
    )

# 2) Read, stack, and take the mean
imgs = [tiff.imread(f).astype(np.float32) for f in files]  # assumes single-channel, same size
mean_img = np.mean(np.stack(imgs, axis=0), axis=0)
sum_img = np.sum(np.stack(imgs, axis=0), axis=0)

# 2) Read, stack, and take the mean
imgs1 = [tiff.imread(f).astype(np.float32) for f in files1]  # assumes single-channel, same size
mean_img1 = np.mean(np.stack(imgs1, axis=0), axis=0)
sum_img1 = np.sum(np.stack(imgs1, axis=0), axis=0)


import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = SCRIPT_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

# ---------- (A) Save each image individually, borderless ----------
plt.imsave(output_dir / "recon_mean_image.png", mean_img, cmap="gist_heat")
plt.imsave(output_dir / "recon_sum_image.png", sum_img, cmap="gist_heat")
plt.imsave(output_dir / "original_mean_image.png", mean_img1, cmap="gist_heat")
plt.imsave(output_dir / "original_sum_image.png", sum_img1, cmap="gist_heat")

# ---------- (B) (optional) Show + save a 2×2 montage (titles visible), no border around figure ----------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
imgs = [
    (mean_img,  "Reconstructed Mean Image"),
    (sum_img,   "Reconstructed Sum Image"),
    (mean_img1, "Original Mean Image"),
    (sum_img1,  "Original Sum Image"),
]
for ax, (arr, title) in zip(axes.ravel(), imgs):
    im = ax.imshow(arr, cmap="gist_heat")
    ax.set_title(title)
    ax.axis("off")

# fill the canvas and save without outer padding
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
fig.savefig(output_dir / "montage_2x2.png", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()
plt.close(fig)
