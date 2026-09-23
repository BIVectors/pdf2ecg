from load_musepdf import load_musepdf
from load_musexml import load_musexml
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Path stuff
# This assumes that the paths are the same as in the pdf2ecg Github repo
# If you change the file/folder structure, this may not work and you
# can just replace pdfFile and xmlFile with the actual paths
here = Path.cwd()                 # .../pdf2ecg/python
repo = here.parent                # .../pdf2ecg
sys.path.insert(0, str(here))     # so load_musepdf imports

# Can manually enter file paths here if needed
pdfFile = repo / "ecgs" / "ecg_12x1.pdf"
xmlFile = repo / "ecgs" / "ecg.xml"

# PDF data
signals, info = load_musepdf(str(pdfFile), "verbose")
II_pdf = signals["II"]
hz_pdf = signals["hz"]

# XML data
hz_xml, I_xml, II_xml, V1_xml, V2_xml, V3_xml, V4_xml, V5_xml, V6_xml = load_musexml(str(xmlFile))

# Show that XML and PDF leads are exactly equal
# This will fail if the leads are of different lengths too
# Be alert for floating point errors as we are comparing mV
exact = np.array_equal(II_pdf, II_xml)
print("PDF and XML leads are equal:", exact)

# Plot XML and PDF leads
plt.figure(figsize=(12, 4))
plt.plot(II_xml, "k-", linewidth=2, label="XML")
plt.plot(II_pdf, "r--", linewidth=1, label="PDF")
plt.xlabel("Samples")
plt.ylabel("mV")
plt.title("Comparison of XML and Reconstructed PDF Waveforms")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
