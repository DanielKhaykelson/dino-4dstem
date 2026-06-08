"""Concordance of two ALREADY-ALIGNED, same-grid label images (no registration).
Usage: python tools/iou_sam_vs_kmap.py sam.PNG kmap.PNG
A = colored (saturated) pixels in sam; B = kmap classes 1(174,199,232)+8(148,103,189)."""
import sys, numpy as np
from PIL import Image
sam = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
kmap = np.asarray(Image.open(sys.argv[2]).convert("RGB")).astype(int)
H = min(sam.shape[0], kmap.shape[0]); W = min(sam.shape[1], kmap.shape[1])
sam, kmap = sam[:H, :W], kmap[:H, :W]                  # crop to common size, NO resize
A = (sam.max(-1) - sam.min(-1)) > 30                   # any saturated colour
def near(img, rgb, tol=20):
    d = img - np.array(rgb); return (d*d).sum(-1) <= tol*tol
B = near(kmap, (174,199,232)) | near(kmap, (148,103,189))
inter = int((A & B).sum()); union = int((A | B).sum())
print(f"IoU={inter/union:.4f}  Dice={2*inter/(int(A.sum())+int(B.sum())):.4f} "
      f"containment(SAM in DINO)={inter/int(A.sum()):.4f} "
      f"containment(DINO in SAM)={inter/int(B.sum()):.4f}")
