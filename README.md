# Image Encryption Based on Grundy Colouring of Random Graphs

Python implementation and experiment scripts for the paper

> M.A.S. Rafi, F. Ahmad, M. Azeem, M.G.A. Malik,
> *Key-dependent Grundy colouring of random graphs for RGB image encryption*.

Every colour component of an RGB image is treated as a vertex of a graph that
joins neighbouring pixels, the three channels of each pixel, and long-range
pairs chosen by the secret key. A key-dependent first-fit (Grundy) colouring of
this graph, computed in parallel with the Jones–Plassmann method, drives a
permutation of the image and two diffusion passes, one from the lowest colour
class to the highest and one back. All key material comes from SHA-256 in
counter mode, and decryption recovers the image exactly.

## Repository contents

| File | Purpose |
|---|---|
| `grundy_image_encryption.py` | The encryption and decryption algorithm and all security metrics. Can be run on a single image. |
| `run_experiments.py` | Reproduces every table and figure of Section 4 of the paper. |
| `get_images.py` | Prepares the seven test images in `images/`. |
| `results/all_results.json` | The numbers reported in the paper. |
| `results/tables.md` | The same numbers as readable tables. |
| `results/figures/` | Plain, cipher and decrypted images, histograms, 3D correlation plots. |

## Requirements

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## Quick start: one image

```bash
python grundy_image_encryption.py path/to/image.png
```

This encrypts and decrypts the image, checks that the recovery is exact, prints
the colouring statistics, MSE/PSNR, key and plaintext sensitivity (NPCR, UACI),
correlation coefficients and entropy, and saves `encrypted.png`,
`decrypted.png` and `correlation_3d.png`. Without an argument it looks for
`images/baboon.png`.

## Reproducing the paper

1. **Test images.** Astronaut and Immunohistochemistry are included in
   scikit-image. The other five come from the
   [USC-SIPI database](https://sipi.usc.edu/database/database.php?volume=misc)
   and are not redistributed here. Download `4.2.03` (Baboon), `4.2.07`
   (Peppers), `4.2.05` (Airplane), `4.2.06` (Sailboat) and `4.1.05` (House),
   put the TIFF files in `images/sipi/`, then run

   ```bash
   python get_images.py
   ```

2. **Experiments.**

   ```bash
   python run_experiments.py              # all seven images
   python run_experiments.py peppers      # or just some of them
   ```

   The results are written to `results/`. With the demonstration key and
   nonce in `grundy_image_encryption.py`, the output matches the tables in the
   paper. Running times depend on the machine (the paper used an Intel Xeon at
   2.10 GHz, about 6–8 s per 512×512 image for three rounds).

## Main results (from `results/all_results.json`)

| Test | Result over the 7 images | Ideal |
|---|---|---|
| Exact recovery | all images | yes |
| Cipher entropy, 512×512 images | 7.9992 – 7.9994 | 8 |
| χ² of cipher histograms | 212.36 – 288.54 (all 21 channels pass) | < 293.25 |
| Largest cipher correlation | 0.036 | 0 |
| NPCR, one-bit key / plaintext change | 99.58 – 99.65 % | 99.6094 % |
| UACI, one-bit key / plaintext change | 33.33 – 33.58 % | 33.4635 % |

## Parameters

`ROUNDS = 3` and `LONG_EDGES = 2` are set at the top of
`grundy_image_encryption.py`, together with the 256-bit demonstration key and
128-bit nonce. For real use, choose a random key and a fresh nonce for every
image; the nonce is not secret but must never be reused with the same key.

## Citation

If you use this code, please cite the paper above.
