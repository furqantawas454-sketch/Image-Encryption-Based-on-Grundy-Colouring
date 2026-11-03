"""
Reproduces every table and figure of Section 4 of the paper.

Usage:
    python get_images.py          # once: prepares the test images in images/
    python run_experiments.py     # runs all tests, writes results/ and figures

Outputs:
    results/all_results.json      raw numbers for every image
    results/tables.md             the tables of the paper in Markdown
    results/figures/*.png         plain/cipher/decrypted images, histograms,
                                  3D correlation scatter plots
"""
import json
import os
import sys
import time
import gc

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import grundy_image_encryption as gce

IMAGES = ["baboon", "peppers", "airplane", "sailboat", "house",
          "astronaut", "immunohistochemistry"]
LABELS = {"baboon": "Baboon", "peppers": "Peppers", "airplane": "Airplane",
          "sailboat": "Sailboat", "house": "House", "astronaut": "Astronaut",
          "immunohistochemistry": "Immunohistochemistry"}
SCATTER = ["baboon", "peppers"]          # 3D correlation plots in the paper
IMG_DIR, OUT_DIR = "images", "results"
FIG_DIR = os.path.join(OUT_DIR, "figures")


def chi_square(channel):
    e = channel.size / 256
    return float(((np.bincount(channel.ravel(), minlength=256) - e) ** 2 / e).sum())


def analyse(name):
    plain = np.array(Image.open(os.path.join(IMG_DIR, f"{name}.png")).convert("RGB"), dtype=np.uint8)
    M, N, _ = plain.shape
    r = {"size": f"{M}x{N}"}

    rd0 = gce.build_round(M, N, gce.KEY, gce.NONCE, 0)
    proper, grundy, k = gce.check_coloring(rd0["color"], rd0["src"], rd0["dst"])
    deg = np.bincount(rd0["src"], minlength=rd0["L"])
    r.update(colours_round1=k, proper=bool(proper), grundy=bool(grundy),
             max_degree=int(deg.max()), avg_degree=float(deg.mean()))
    del rd0
    gc.collect()

    t0 = time.time(); C = gce.encrypt_image(plain); r["enc_time_s"] = time.time() - t0
    t0 = time.time(); D = gce.decrypt_image(C); r["dec_time_s"] = time.time() - t0
    r["exact_recovery"] = bool(np.array_equal(plain, D))
    r["mse"], r["psnr_db"] = map(float, gce.calc_mse_psnr(plain, C))

    key2 = bytearray(gce.KEY); key2[-1] ^= 1          # one-bit key change
    Ck = gce.encrypt_image(plain, key=bytes(key2))
    r["key_npcr_uaci"] = [list(map(float, x)) for x in gce.calc_npcr_uaci_rgb(C, Ck)]
    Wk = gce.decrypt_image(C, key=bytes(key2))
    r["wrong_key_npcr_vs_plain"] = [float(x[0]) for x in gce.calc_npcr_uaci_rgb(plain, Wk)]

    p2 = plain.copy(); p2[0, 0, 0] ^= 1                # one-bit plaintext change
    Cp = gce.encrypt_image(p2)
    r["plain_npcr_uaci"] = [list(map(float, x)) for x in gce.calc_npcr_uaci_rgb(C, Cp)]

    r["correlation"] = {}
    for d in "hvd":
        r["correlation"][d] = []
        for c in range(3):
            xp, yp = gce.get_adjacent_pairs(plain, c, d)
            xc, yc = gce.get_adjacent_pairs(C, c, d)
            r["correlation"][d].append([gce.correlation_coefficient(xp, yp),
                                        gce.correlation_coefficient(xc, yc)])
    r["entropy_plain"] = gce.calc_entropy(plain)
    r["entropy_cipher"] = gce.calc_entropy(C)
    r["chi2_cipher"] = [chi_square(C[:, :, c]) for c in range(3)]

    for tag, im in (("plain", plain), ("cipher", C), ("decrypted", D)):
        Image.fromarray(im).save(os.path.join(FIG_DIR, f"{name}_{tag}.png"))
    if name in SCATTER:
        fig = gce.plot_correlation_3d(plain, C, samples=3000,
                                      save_path=os.path.join(FIG_DIR, f"correlation_3d_{name}.png"))
        plt.close(fig)
    return r


def write_tables(R):
    ch = "RGB"
    L = ["# Results\n", "## Entropy and chi-square (critical value 293.25)\n",
         "| Image | Ch. | Entropy plain | Entropy cipher | Chi-square cipher |", "|---|---|---|---|---|"]
    for n, r in R.items():
        for c in range(3):
            L.append(f"| {LABELS[n]} | {ch[c]} | {r['entropy_plain'][c]:.4f} | "
                     f"{r['entropy_cipher'][c]:.4f} | {r['chi2_cipher'][c]:.2f} |")
    L += ["\n## Correlation of adjacent pixels (plain / cipher)\n",
          "| Image | Ch. | Horizontal | Vertical | Diagonal |", "|---|---|---|---|---|"]
    for n, r in R.items():
        for c in range(3):
            cells = " | ".join(f"{r['correlation'][d][c][0]:.4f} / {r['correlation'][d][c][1]:.4f}" for d in "hvd")
            L.append(f"| {LABELS[n]} | {ch[c]} | {cells} |")
    L += ["\n## NPCR / UACI (%)\n",
          "| Image | Ch. | Key NPCR | Key UACI | Plaintext NPCR | Plaintext UACI |", "|---|---|---|---|---|---|"]
    for n, r in R.items():
        for c in range(3):
            k, p = r["key_npcr_uaci"][c], r["plain_npcr_uaci"][c]
            L.append(f"| {LABELS[n]} | {ch[c]} | {k[0]:.4f} | {k[1]:.4f} | {p[0]:.4f} | {p[1]:.4f} |")
    L += ["\n## Colouring, timing, MSE, PSNR\n",
          "| Image | Size | Colours | Max deg. | Enc. (s) | Dec. (s) | MSE | PSNR (dB) | Exact |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in R.items():
        L.append(f"| {LABELS[n]} | {r['size']} | {r['colours_round1']} | {r['max_degree']} | "
                 f"{r['enc_time_s']:.2f} | {r['dec_time_s']:.2f} | {r['mse']:.2f} | "
                 f"{r['psnr_db']:.4f} | {r['exact_recovery']} |")
    open(os.path.join(OUT_DIR, "tables.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    names = sys.argv[1:] or IMAGES
    missing = [n for n in names if not os.path.isfile(os.path.join(IMG_DIR, f"{n}.png"))]
    if missing:
        sys.exit(f"Missing images {missing}. Run `python get_images.py` first.")
    R = {}
    for n in names:
        print(f"Analysing {n} ...", flush=True)
        R[n] = analyse(n)
        print(f"  exact recovery: {R[n]['exact_recovery']}, "
              f"time {R[n]['enc_time_s']:.1f} s / {R[n]['dec_time_s']:.1f} s", flush=True)
    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(R, f, indent=1)
    write_tables(R)
    print(f"Done. See {OUT_DIR}/tables.md and {FIG_DIR}/")
