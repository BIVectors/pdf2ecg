"""
*** load_musexml.py
Python port of load_musexml.m. Returns the sampling rate and the eight independent leads. 
Leads III, aVR, aVL and aVF can be derived from I and II.

This code is adapted from load_musexml.m from BRAVEHEART ECG/VCG Software:
https://github.com/BIVectors/BRAVEHEART

Copyright 2026: Jonathan W. Waks
Source code available at: https://github.com/BIVectors/pdf2ecg
Contact: braveheart.ecg@gmail.com

pdf2ecg is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version.

pdf2ecg is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. 
If not, see <https://www.gnu.org/licenses/>.

This software is for research purposes only and is not intended to diagnose or treat any disease.

"""

import base64
import sys
import xml.etree.ElementTree as ET

import numpy as np


def _elget(el, name):
    """Text of the first element with the given tag name."""
    for child in el.iter():
        tag = child.tag.split("}")[-1]  # ignore any XML namespace
        if tag == name:
            return (child.text or "").strip()
    raise ValueError(f"element <{name}> not found")


def _elgetn(el, name):
    return float(_elget(el, name))


def load_musexml(filename, waveform_type="Rhythm"):
    """Load ECG leads from a MUSE XML file.

    Input:
    filename : str -- MUSE XML file path
    waveform_type : str, optional -- which waveform to read: "Rhythm" (default) or "Median".

    Output:
    hz : float -- Sampling frequency in Hz.
    I, II, V1, V2, V3, V4, V5, V6 : numpy.ndarray -- Lead signals in millivolts.
    """
    tree = ET.parse(filename)  # DTDs are not loaded by ElementTree
    root = tree.getroot()

    waveform = None
    for w in root.iter():
        if w.tag.split("}")[-1] != "Waveform":
            continue
        if _elget(w, "WaveformType") == waveform_type:
            waveform = w
            break
    if waveform is None:
        raise ValueError(f"{filename}: no '{waveform_type}' waveform found")

    hz = _elgetn(waveform, "SampleBase")
    exponent = _elgetn(waveform, "SampleExponent")
    if exponent != 0:
        raise ValueError(f"{filename}: nonzero sample exponent {exponent}")

    signals = {}
    for lead in waveform.iter():
        if lead.tag.split("}")[-1] != "LeadData":
            continue

        gain = _elgetn(lead, "LeadAmplitudeUnitsPerBit")
        unit = _elget(lead, "LeadAmplitudeUnits")
        if unit != "MICROVOLTS":
            raise ValueError(f"{filename}: expected MICROVOLTS but found {unit}")

        lead_id = _elget(lead, "LeadID")

        offset = _elgetn(lead, "LeadOffsetFirstSample")
        if offset != 0:
            raise ValueError(
                f"{filename}: lead {lead_id} with {offset:g} bytes of invalid data"
            )

        baseline = _elgetn(lead, "FirstSampleBaseline")
        bytes_per_sample = _elgetn(lead, "LeadSampleSize")
        if bytes_per_sample != 2:
            raise ValueError(
                f"{filename}: expected 2 bytes per sample but found {bytes_per_sample:g}"
            )

        raw = base64.b64decode(_elget(lead, "WaveFormData"))
        # data is little-endian per MUSE spec
        intsignal = np.frombuffer(raw, dtype="<i2").astype(float)
        intsignal = intsignal + baseline
        signals[lead_id] = intsignal * gain / 1000.0

    order = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    missing = [name for name in order if name not in signals]
    if missing:
        raise ValueError(f"{filename}: missing leads {', '.join(missing)}")

    return (hz, *[signals[name] for name in order])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python load_musexml.py <file.xml>")
    hz, I, II, V1, V2, V3, V4, V5, V6 = load_musexml(sys.argv[1])