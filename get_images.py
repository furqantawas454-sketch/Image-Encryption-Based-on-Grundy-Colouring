"""
Prepares the seven test images used in the paper as PNG files in images/.

* Astronaut and Immunohistochemistry are shipped with scikit-image.
* Baboon, Peppers, Airplane, Sailboat and House come from the USC-SIPI image
  database (volume 3, miscellaneous). They are not redistributed here. Download
  the TIFF files 4.2.03, 4.2.07, 4.2.05, 4.2.06 and 4.1.05 from
  https://sipi.usc.edu/database/database.php?volume=misc
  and put them in images/sipi/. This script converts them to PNG.
"""
import os
import sys

from PIL import Image
from skimage import data

SIPI = {"baboon": "4.2.03", "peppers": "4.2.07", "airplane": "4.2.05",
        "sailboat": "4.2.06", "house": "4.1.05"}
OUT = "images"
SIPI_DIR = os.path.join(OUT, "sipi")

os.makedirs(SIPI_DIR, exist_ok=True)
Image.fromarray(data.astronaut()).save(os.path.join(OUT, "astronaut.png"))
Image.fromarray(data.immunohistochemistry()).save(os.path.join(OUT, "immunohistochemistry.png"))
print("scikit-image: astronaut.png, immunohistochemistry.png")

missing = []
for name, code in SIPI.items():
    candidates = [os.path.join(SIPI_DIR, f"{code}{ext}") for ext in (".tiff", ".tif", ".png")]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        missing.append(code)
        continue
    Image.open(path).convert("RGB").save(os.path.join(OUT, f"{name}.png"))
    print(f"SIPI {code} -> {name}.png")

if missing:
    print("\nStill missing SIPI files:", ", ".join(missing))
    print("Download them from https://sipi.usc.edu/database/database.php?volume=misc")
    print(f"and place them in {SIPI_DIR}/, then run this script again.")
    sys.exit(1)
