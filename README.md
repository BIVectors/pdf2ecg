# Bit-Exact Reconstruction and Certification of Electrocardiogram Waveforms from Vector-Encoded PDF Files

## Reconstruct and certify bit-exact waveforms from GE MUSE 12-lead ECG PDFs.   

[![badge](https://badgen.net/badge/MATLAB/R2022a/?color=green)](https://www.mathworks.com/products/matlab.html)
[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=BIVectors/pdf2ecg&file=matlab/pdf2ecgdemo.m)   
[![badge](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)](https://www.python.org/) ![badge](https://img.shields.io/badge/NumPy-2.4-blue?logo=numpy&logoColor=white) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BIVectors/pdf2ecg/HEAD?labpath=python/pdf2ecgdemo.ipynb)   
![badge](https://badgen.net/badge/License/GPL-3.0/?color=red)    
[![badge](https://img.shields.io/badge/📄-Methods%20Manuscript-F7DF1E?style=flat)](https://medrxiv.org)

**Jonathan W. Waks, MD   
Harvard-Thorndike Electrophysiology Institute, Department of Cardiovascular Medicine,   
Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA   
braveheart.ecg@gmail.com**

---
<img width="1398" height="433" alt="image" src="https://github.com/user-attachments/assets/32374727-4d02-4265-b729-53f4b93e84a4" />

<img width="1398" height="433" alt="image" src="https://github.com/user-attachments/assets/2b9d42cb-ac77-40f9-95b5-0ff1377e4f9e" />


---

`load_musepdf` parses a GE MUSE 12-lead ECG PDF generated with vector graphics, 
extracts the polylines that make up the waveforms, and inverts the
double quantization introduced when the device rendered the ECG for print. The
result is a bit-exact reconstruction of the original waveforms which were used
to generate the PDF rather than an approximation.  This allows use of PDF ECGs in AI-ECG or vectorcardiographic
analyses without needing to worry that small errors in signal digitization could cause unpredictable 
downstream errors when compared to use of raw data formats such as XML or DICOM.

As the parameters that are used for reconstruction (such as the stream unit to analogue-to-digital [ADU] ratio or baseline zero voltage references) 
are not directly encoded in the PDF (unlike raw digital formats such as XML or DICOM), and must be extracted or presumed,
the software leverages the double quantization of PDF stream unit values to provide
certification that bit-exact reconstruction was successful.  This is critically important to avoid 
propagating silent errors into downstream analyses when processing PDFs from unknown sources.

The software also identifies signal clipping, where part of the waveform is not rendered on the PDF
due to being outside of the rendering area.  This results in loss of data which cannot be recovered.
The ECGs flagged for clipping should be inspected prior to use as they can contain significant reconstruction
errors.

Please see our preprint at XXX for a full discussion of the software methods and results of testing
on a large database of PDF ECGs in multiple formats.

---

## Supported formats

| Property| Supported |
|:---|:---|
| Source | GE MUSE-generated PDFs only |
| Sweep speed | 25 mm/s |
| Gain | 5, 10 and 20 mm/mV, uniform across all leads |
| Layouts | 12×1 rhythm; 4×3 2.5 s with 0, 1 or 3 rhythm strips; 6×2 5 s |
| Pages | 1, or 2 where the second page carries no waveforms |
| Compression | FlateDecode with a direct `/Length` |

**Not supported**: 50 mm/s and 12.5 mm/s sweep speeds, mixed limb/precordial gains,
non-MUSE PDFs, and PDFs where the waveform streams use an indirect `/Length`
reference. Each of these is detected and raises a descriptive error rather than
returning wrong data.

**Only the 12×1 layout aligns all leads in time.** In 4×3 and 6×2 formats the
leads are recorded in different time windows, which matters if you intend to
build median beats.

---

## Requirements

Two implementations are provided and produce identical output:

|Implementation|Requirements|
|:---|:---|
| `matlab/load_musepdf.m` | MATLAB R2022a or later |
| `python/load_musepdf.py` | Tested with Python 3.14.3 with numpy 2.4.5 |

---

## Usage
Both implementations require input of a PDF filename as a string followed by various options.   

Sample PDF and matching XML files are provided to test the software. XML files can be read using `load_musexml.m` and `load_musexml.py`. 

`pdf2ecgdemo.m` and `pdf2ecgdemo.py` can be run to load an ECG in PDF and XML formats and then compare the outputs.
 

### Options

| Option | Effect |
|:---|:---|
| `verbose` | Print parsing, layout, calibration, and certification detail|
| `adu` | Return signals in integer ADUs instead of mV |
| `exportfile` | Write the processed ECG data to `<basename>_digitized.csv` (or `_digitized_adu.csv`) |
| `calcleads` | **(default)** Extracts leads I, II and V1–V6; computes leads III, aVR, aVL and aVF from leads I and II |
| `pdfleads` | Extract all leads directly from the PDF |


### MATLAB
[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=BIVectors/pdf2ecg&file=matlab/pdf2ecgdemo.m)

```matlab
% Process PDF files with different options
[signals, info] = load_musepdf('ecg.pdf');
[signals, info] = load_musepdf('ecg.pdf', 'verbose');
[signals, info] = load_musepdf('ecg.pdf', 'adu', 'pdfleads', 'exportfile');

% Report total number of clipped samples
% If 0 then there is no clipping
signals.totalClipping

% Report if ECG passed certification tests
% If 1 then passed
signals.certified

```
```matlab
% PDF data
[signals, info] = load_musepdf('ecg.pdf','verbose');
II_pdf = signals.II;

% XML data
[hz, I, II, V1, V2, V3, V4, V5, V6] = load_musexml('ecg.xml');
II_xml = II';    % Convert to column vector

% Show that XML and PDF leads are exactly equal
% Be alert for floating point errors as we are comparing mV
exact = isequal(II_pdf,II_xml);
fprintf('PDF and XML leads are equal: %s\n', string(exact));

% Plot XML and PDF leads
figure
hold on
title('Comparison of XML and Reconstructed PDF Waveforms')
plot(II_xml, 'k', 'LineWidth', 2 ,'DisplayName','XML')
plot(II_pdf, 'r--', 'LineWidth', 1 ,'DisplayName','PDF')
xlabel('Samples')
ylabel('mV')
legend
```


### Python
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BIVectors/pdf2ecg/HEAD?labpath=python/pdf2ecgdemo.ipynb)

```python

# Process PDF files with different options
from load_musepdf import load_musepdf

signals, info = load_musepdf("ecg.pdf")
signals, info = load_musepdf("ecg.pdf", "verbose")
signals, info = load_musepdf("ecg.pdf", "adu", "pdfleads", "exportfile")

# Report total number of clipped samples
# If 0 then there is no clipping
print(signals["totalClipping"])

# Report if ECG passed certification tests
# If 1 then passed
print(signals["certified"])

```

```python
from load_musepdf import load_musepdf
from load_musexml import load_musexml
import numpy as np
import matplotlib.pyplot as plt

# PDF data
signals, info = load_musepdf("ecg.pdf",'verbose')
II_pdf = signals["II"]
hz_pdf = signals["hz"]

# XML data
hz_xml, I_xml, II_xml, V1_xml, V2_xml, V3_xml, V4_xml, V5_xml, V6_xml = load_musexml("ecg.xml")

# Show that XML and PDF leads are exactly equal
# This will fail if the leads are of different lengths too
exact = np.array_equal(II_pdf, II_xml)
print("PDF and XML leads are equal:", exact)

# Plot XML and PDF leads
plt.figure(figsize=(12, 4))
plt.plot(II_xml, "k-", linewidth=2, label="XML")
plt.plot(II_pdf, "r--", linewidth=1, label="PDF")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.title("Comparison of XML and Reconstructed PDF Waveforms")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

```

There is also a command line entry point:

```
python load_musepdf.py ecg.pdf verbose exportfile
```

Where MATLAB returns structs, Python returns dictionaries with the same field
names. Lead arrays are 1-D numpy arrays.



#### calcleads vs pdfleads

MUSE computes leads III, aVR, aVL, and aVF from leads I and II internally, rounds them to integer ADU values, and 
these ADU values are then used to generate the PDF. `pdfleads` recovers exactly those values. 
`calcleads` instead calculates leads III, aVR, aVL, and aVF from leads I and II, resulting in values that can contain
half-integer ADU values due to division by 2.

In `calcleads` mode the derived leads inherit the clipping and certification of leads I and
II.  If both source leads are bit-exact then anything derived from them is too, and if either lead 
has clipping, the derived leads also inherit that error.

---

## Output

### `signals`

| Field | Contents |
|:---|:---|
| `I`, `II`, `III` … `V6` | Lead data, mV by default or ADU with the `adu` option |
| `hz` | Sampling frequency |
| `units` | Amplitude units of the lead signals in `'mV'` or `'ADU'` |
| `totalClipping` | Total clipped samples across all 12 leads |
| `certified` | `1` pass, `0` fail, `NaN` indeterminate |
| ***Optional fields if rhythm strip is present*** | |
| `rhythm` | Rhythm strip leads, for layouts that contain them, associated with a field of its name (e.g. `rhythm.II`) |
| `totalClippingRhythm` | Total clipped samples across all rhythm strip leads |


### `info`

| Field | Contents |
|:------|:------------|
| `hz` | Sampling rate in Hz. |
| `leadOrder` | Lead names in the order used throughout the output. |
| `leadsMode` | Mode for augmented leads (`'calcleads'` or `'pdfleads'`). |
| `ecgFormat` | Page layout of the source ECG (e.g. `'12×1'`). |
| `ecgFormatStr` | Additional description of the PDF layout. |
| `numPolylines` | Total number of vector polylines extracted. |
| `numLeadsDetected` | Number of ECG leads detected. |
| `cmMatrix` | Affine transform mapping source coordinates to centimeters. |
| `certified` | Flag for if passed (`1`) or failed (`0`) certification. |
| `diagnostics` | Structure containing results of validation tests. (see below) |
| `lowPassFreq` | Low-pass filter cut-off frequency in Hz. |
| `XMLmicrovoltsLSB` | μV per ADU (value used is assumed from MUSE). |
| `unitsPerMv` | Stream units per millivolt. |
| `mmPerMv` | ECG amplitude gain (mm per mV). |
| `mmPerSec` | ECG sweep speed (mm per second). |
| `alpha` | Stream-to-ADU conversion, α. |
| `offsets` | Per-lead baseline offset for each lead in stream units. |
| `numPages` | Number of pages in the source PDF. |
| `numStreams` | Number of data streams. |
| `clipping` | Nested structure with per-lead clipping details.  For each lead includes total number of clipped samples and which samples are clipped.  Includes rhythm strips if present. |
| `maxStreamUnits` | Per-lead maximum value in stream units. |
| `minStreamUnits` | Per-lead minimum value in stream units. |
| ***Optional fields if rhythm strip is present*** | |
| `rhythmStripLeads` | Rhythm strip lead names. |


### Diagnostics

The `info.diagnostics` structure contains information on certification test results for each lead.  
The test results are explained in detail in the methods manuscript.

| Field | Contents |
|:---|:---|
| `certified` | Overall certification result: passed (`1`), failed (`0`), or indeterminate (`NaN`).  |
| `W` | Result of the residue width test. Note: calculated leads return `NaN` |
| `WMargin` | How much of the expected residue arc is filled.  `0` is completely filled.  Positive values indicate the arc is not completely filled, and negative values indicate values outside the arc (test failed).|
| `passWLimit` | The maximum value of `W` to pass the residue width test.  `W > passWLimit` fails.|
| `passW` | Result of the residue width (`W`) test: passed (`1`), failed (`0`), or indeterminate (`NaN`) |
| `GCD` | Greatest common divisor of the stream unit values.  Values > 1 fail. |
| `numD` | Number of distinct differences used in the GCD test. |
| `passGCD` | Result of the GCD test: passed (`1`), failed (`0`), or indeterminate (`NaN`)|
| `BOut` | Number of residues falling outside the expected window.  Values > 0 fail |
| `BOutClasses` | Number of distinct residue classes among those outliers in `BOut` |
| `passB` | Result of the baseline test: passed (`1`), failed (`0`), or indeterminate (`NaN`).|
| ***Optional fields if in*** `calcleads` ***mode*** | |
| `calcleadsOriginal` | The above values prior to any adjustment of leads III, aVR, aVL, and aVF due to `calcleads` mode.  In `calcleads` mode leads III, aVR, aVL, and aVF inherit certification from leads I and II.|
| ***Optional fields if rhythm strip is present*** | |
| `rhythm` | Same data as above for rhythm strips. |

---

## Important Information

- Always check `certified` before using the output quantitatively. A value of
  `0` means at least one lead is provably not bit-exact, most often because of
  clipping.
- Clipping is flagged from a single sample crossing a set rendering threshold. There is no way to tell
  whether such a sample merely touched the limit or was cut off far above it, and the error due to clipping therefore
  cannot be determined from the PDF alone.
- `4.88` µV/LSB is assumed, matching the MUSE XML. Due to the certification tests, a device configured differently would fail certification rather than silently return incorrect data.

---

## Citation

If you use this software, please cite:

> *(methods paper — add citation here)*

---
## License

Copyright 2026 Jonathan W. Waks
All rights reserved.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/ or the LICENSE file included in this repository.

