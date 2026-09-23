"""
*** PDF2ECG
Convert MUSE Encoded PDF ECG with Vector Graphics into bit-exact waveforms 
and certify that waveform reconstruction is bit-exact

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

Adobe PDF specification is available at: https://opensource.adobe.com/dc-acrobat-sdk-docs/standards/pdfstandards/pdf/PDF32000_2008.pdf

INPUTS:
1st input must be the filename of the .pdf as a string

Options to include as strings after filename include:

'verbose' :: will print information on signal parsing/results/signal clipping when the function runs.  
        Default: If this string is not specified the function returns no text information.
 
'pdfleads' :: Extract all leads directly from the .pdf file

'calcleads' :: Extract leads I, II, and V1-V6 from the .pdf file, but calculate leads III, aVR, aVL, and aVF 
        from the extracted values of leads I and II. 

If 'pdfleads', 'calcleads', is not specified, the default is to use 'calcleads' 

'adu' :: Data is output in ADU values and not physical units of mV

'exportfile' :: will export the data from the .pdf into a .csv with the filename 'basename_digitized.csv'
        if output is in mV or 'basename_digitized_adu.csv' if output is in ADU units. 
        Default: If this string is not specified the file is not exported to .csv

OUTPUTS:
signals :: a structure containing the lead data in mV (signals["I"], signals["V1"], etc), the total number of clipped 
        points in the full ECG (signals["totalClipping"]), sampling frequency (signals["hz"]), any rhythm strips if not in a 12 
        rhythm strip format (signals["rhythm"]), and if any rhythm strips have clipping (signals["totalClippingRhythm"])

info :: Layout, calibration, and per-lead certification diagnostics.

*Note*: this function as currently only been tested and verified using GE MUSE format .pdf files with sweep 
speeds are 25 mm/sec.  Variable voltage gains (5 mm/mV, 10 mm/mV, and 20 mm/mV) have been tested/validated.  
12 rhythm strips, 3x4 2.5 sec with 0-3 rhythm strups, and 6x2 5 sec with 0-3 rhythm strips are supported,
but only the 12 rhythm strip format will allow all beats to temporally align in time.  This temporal alignment 
is CRITICAL for accurate median beat generation.  

*Requirements*: Python 3, numpy

    
USAGE:
    from load_musepdf import load_musepdf

    signals, info = load_musepdf('ecg.pdf', 'verbose')
    signals, info = load_musepdf('ecg.pdf', 'adu', 'pdfleads', 'exportfile')

    II_pdf = signals["II"]
    hz_pdf = signals["hz"]
"""


# Import libraries 
import numpy as np
import math
import os
import re
import zlib
from fractions import Fraction

# Prevent leakage of public names from other libraries
__all__ = ["load_musepdf", "LoadMusePdfError"]

# Setup error class
class LoadMusePdfError(Exception):
    pass

# Declare constants
# Lead order
lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Options for function
VALID_OPTIONS = {"exportfile", "verbose", "calcleads", "pdfleads", "adu", "stream"}

# Clipping limits in stream units
MIN_CLIP_VAL = 450
MAX_CLIP_VAL = 21150

# Microvolts per LSB, obtained from XML.  If this is incorrect the certification tests will fire
MUV_PER_LSB = 4.88


#######################################################################################
# MATLAB comparable helper functions since the exact function that MATLAB uses are not
# available in Python

def _mround(x):
    # MATLAB round(): half away from zero (numpy rounds half to even).  
    # This shouldnt really matter since there are no exact rounds as noted 
    # in the methods, but will implement this for consistency
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def _isnan(v) -> bool:
    # Determine if is NaN
    return isinstance(v, float) and math.isnan(v)

def _mode(x: np.ndarray):
    # MATLAB mode(): most frequent value, but if there is a tie smallest wins
    vals, counts = np.unique(x, return_counts=True)
    mostfreq = counts.max()
    return vals[counts == mostfreq].min()

def _rat(x: float, tol: float = 1e-12):
    # MATLAB rat(): smallest rational approximation within specified tolerance
    # Set the range of demoninator from 10^1 to 10^12
    for lim in (10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9, 10 ** 10, 10 ** 11, 10 ** 12, 10 ** 13, 10 ** 14, 10 ** 15):
        f = Fraction(x).limit_denominator(lim)
        if abs(float(f) - x) <= tol:
            return f.numerator, f.denominator

def _as_int(x, numb: str) -> int:
    # Float to int, errors like MATLAB's GCD would for a non-integer
    xf = float(x)
    if not float(xf).is_integer():
        raise LoadMusePdfError(f"load_musepdf: {numb} is not an integer ({xf!r})")
    return int(round(xf))

def _sortrows_asc_desc(col1: np.ndarray, col2: np.ndarray) -> np.ndarray:
    # MATLAB sortrows([c1 c2], [1 -2]): c1 ascending, c2 descending
    n = len(col1)
    return np.array(sorted(range(n), key=lambda i: (col1[i], -col2[i])), dtype=int)

def _inflate_bytes(deflated: bytes) -> bytes:
    # zlib-inflate
    try:
        return zlib.decompress(deflated)
    except zlib.error:
        # Retry tolerating a truncated / raw-deflate stream, keeping whatever
        # bytes were recovered before the error, like the Java stream copier.
        for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                d = zlib.decompressobj(wbits)
                out = d.decompress(deflated)
                out += d.flush()
                if out:
                    return out
            except zlib.error:
                continue
        return b""

# Clipping detection
def _find_plateaus(y: np.ndarray, clip_val: float, min_run_len: int) -> np.ndarray:
    # Mask of samples with values >= min_run_len that equal clip_val
    eq = (y == clip_val)
    if min_run_len <= 1:
        return eq

    mask = np.zeros(y.shape, dtype=bool)
    padded = np.concatenate(([0], eq.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    for start, stop in zip(edges[0::2], edges[1::2]):
        if (stop - start) >= min_run_len:
            mask[start:stop] = True
    return mask


def _detect_clipping(stream: np.ndarray, min_run_len: int = 1) -> np.ndarray:
    # Flag clipped samples in a polyline.  A clipped sample is one inside a
    # run of >= minRunLen consecutive identical stream values at either the
    # max or min of the polyline.  This produces bit-exact plateaus which should
    # not happen with real data.

    clip_mask = np.zeros(stream.shape, dtype=bool)
    if stream.size < min_run_len:
        return clip_mask

    max_y = float(stream.max())
    min_y = float(stream.min())
    if max_y == min_y:
        return clip_mask

    if max_y >= MAX_CLIP_VAL or min_y <= MIN_CLIP_VAL:
        clip_mask |= _find_plateaus(stream, MAX_CLIP_VAL, min_run_len)
        clip_mask |= _find_plateaus(stream, MIN_CLIP_VAL, min_run_len)
    return clip_mask


# stream2adu conversion and certification tests
def _stream2adu(stream_y: np.ndarray, alpha: float, p: int, q: int,
                offsets: np.ndarray, lead_idx: int):
    
    # Certification (all three must pass):
    #  1) Residue width, W  - catches incorrect alpha (other than a submultiple)
    #  2) GCD - catches if alpha is an integer submultiple of true value
    #  3) Baseline - catches incorrect baseline offset

    d = {}

    adu = _mround((stream_y - offsets[lead_idx]) / alpha)

    # Residue width test catches if alpha is correct (other than submultiple -- see GCD test)
    # By definition of mod: S mod(p/q) = qS mod(p).
    # and we have alpha = p/q
    # Therefore S mod(alpha) = qS mod(p) which is an integer between 0 and p-1
    # To avoid floating point issues we prefer to use mod(p) since p is an
    # integer but alpha is not

    R = np.mod(q * stream_y, p)
    Ru = np.unique(R)

    r_gaps = np.concatenate([np.diff(Ru), [Ru[0] + p - Ru[-1]]])
    w_ticks = p - float(r_gaps.max())
    W = w_ticks / q

    # Similar to passing when W <= (q-1)/q, using Wticks we get passing when Wticks <= q-1
    # If q is odd this works fine, if q is even theoretically have to know
    # about the rounding convention for values that are exactly half units,
    # so use the slightly looser cenvention of W < 1 (or Wticks < q)

    d["W"] = W
    q_lim = (q - 1) + (1 if q % 2 == 0 else 0)
    d["pass_W"] = bool(w_ticks <= q_lim)
    d["pass_W_limit"] = q_lim / q
    d["Wmargin"] = (q_lim - w_ticks) / q          # stream units; negative fails

    if q == 1:
        #  Integer alpha: no spread to measure.  This never shows up in  MUSE, but if it did 
        #  at some point would likely want to remove the W test from the certification.  
        #  This note is to mention this possibility for furture MUSE or if this code is adapted for
        #  another PDF manufacturer who uses integer alpha
        d["pass_W"] = float("nan")
    
    elif Ru.size < 2:        
        #  Flat lead: insufficient evidence
        #  The actual cutoff used here might warrant change at some point 
         d["pass_W"] = float("nan")  

    # GCD test catches if alpha is an integer submultiple of the true alpha 
    # (missed by arc width since it just wraps around mod alpha)
    # Take non-zero values of A_n+1 - A_n

    diffs = np.abs(np.diff(adu))
    diffs = diffs[diffs != 0]
    d["numD"] = int(diffs.size)

    if diffs.size < 50:                           # flat / disconnected lead
        d["pass_gcd"] = float("nan")
        d["gcd"] = float("nan")
    else:
        g = 0
        for v in diffs:
            g = math.gcd(g, int(v))
            if g == 1:
                break
        d["gcd"] = g
        d["pass_gcd"] = bool(g == 1)


    # Baseline offset test:
    # If alpha is wrong, then it does not make sense to calculate the test of
    # if the offset (B) is correct, becuase if alpha is wrong then this will always show
    # B is wrong.  Additionally, if alpha is wrong, the correct window that
    # the residues need to live in is also not clearly defined - best to just
    # not calculate it as the ECG will be flagged by failing W anyway

    if  d["pass_W"] is False:
        d["pass_B"] = float("nan")
        d["B_out"] = float("nan")
        d["B_out_classes"] = float("nan")
    else:
        n_win = q + (1 if q % 2 == 0 else 0)      # odd q: q ticks, even q: q+1
        n_lo = math.floor(q * offsets[lead_idx] - q / 2) + 1 - (1 if q % 2 == 0 else 0)
        B_window = np.mod(n_lo + np.arange(n_win), p)

        inside = np.isin(R, B_window)
        d["B_out"] = int(np.count_nonzero(~inside))
        d["B_out_classes"] = int(np.unique(R[~inside]).size)
        d["pass_B"] = bool(d["B_out"] == 0)

    # Overall
    tests = [d["pass_W"], d["pass_gcd"], d["pass_B"]]
    if any((not _isnan(t)) and float(t) == 0 for t in tests):
        d["certified"] = 0
    elif any(_isnan(t) for t in tests):
        d["certified"] = float("nan")
    else:
        d["certified"] = 1

    return adu, d


# CSV export
def _pdfsig2csv(signals: dict, pdf_file: str, adu: bool) -> str:
    """Write the 12 leads to <basename>_digitized[_adu].csv (no rhythm strips)."""
    E = np.column_stack([signals[ld] for ld in lead_order])
    base = os.path.splitext(pdf_file)[0]
    filename = base + ("_digitized_adu.csv" if adu else "_digitized.csv")
    np.savetxt(filename, E, delimiter=",", fmt="%.15g")
    return filename


#######################################################################################
# Start main function

def load_musepdf(pdf_file: str, *options: str):
    # -- options ------------------------------------------------------------
    opts = [str(o).lower() for o in options]
    bad = sorted(set(opts) - VALID_OPTIONS)
    if bad:
        raise ValueError(
            "load_musepdf: unknown option(s) %s; valid options are %s"
            % (", ".join(bad), ", ".join(sorted(VALID_OPTIONS)))
        )

    exportfile = "exportfile" in opts
    verbose = "verbose" in opts
    outputadu = "adu" in opts

    leads_mode = "calc"    # 'calc' | 'pdf'; defaults to 'calcl
    for o in opts:                                
        if o == "calcleads":
            leads_mode = "calc"
        elif o == "pdfleads":
            leads_mode = "pdf"

    def verbosePrint(fmt, *args):
        if verbose:
            print(fmt % args if args else fmt, end="")

    if outputadu:
        verbosePrint("Output in ADU units\n")
    if leads_mode == "calc":
        verbosePrint("\nLeads aVR, aVL, and aVF will be calculated from leads I and II\n")
    else:
        verbosePrint("\nAll Leads will be extracted from .pdf file\n")

    # Read PDF file
    if not os.path.isfile(pdf_file):
        raise LoadMusePdfError("load_musepdf: PDF file not found: %s" % pdf_file)

    with open(pdf_file, "rb") as fid:
        raw = fid.read()

    if not raw:
        raise LoadMusePdfError("load_musepdf: File read as empty (0 bytes).")

    raw_str = raw.decode("latin-1")             

    # Check that MUSE was used to generate the PDF because the code may not work for other manufacturers
    if re.search(r"/(?:Producer|Creator)\s*\(MUSE", raw_str) is None:
        raise LoadMusePdfError("load_musepdf: PDF does not appear to be from MUSE.")


    # Count number of pages in the PDF.  In general we only want to parse
    # single page PDFs as this is standard, and anything on the second page
    # would be non-standard leads.  The exceptions to this are standard 12-lead
    # ECGs with a very long physician interpretation, where a second "blank"
    # page with no signals and the rest of the physician interpretation can be
    # present, and if a 12-lead rhythm strip ECG is at 50 mm/s where 2 pages
    # are needed to show the full 10 seconds.  We will support the 2 pages
    # where no signals are on the second page, but will NOT support ECGs at 
    # 50 mm/s because this is very rare and adds significant complexity to
    # parsing the leads over 2 pages.  If needed may add this functionality in
    # a future release.

    # We therefore allow 1 or 2 page PDFs.  MUSE uses PDF 1.5 compressed object 
    # streams, so "N 0 obj" headers aren't reliably present as plain text and we 
    # can't trace streams back to objects that way.  Instead we use byte position:
    # in MUSE PDFs each page's content stream sits immediately after its page
    # dictionary, so we cut off at the start of page 2's dictionary.

    page_starts = [m.start() for m in re.finditer(r"/Type\s*/Page(?![a-zA-Z])", raw_str)]
    n_pages = len(page_starts)
    if n_pages < 1 or n_pages > 2:
        raise LoadMusePdfError(
            "load_musepdf: PDF has %d pages. Only 1- or 2-page ECG PDFs are supported."
            % n_pages
        )

    # MUSE uses PDF 1.5 compressed object streams, so "N 0 obj" headers are not
    # reliably plain text; instead cut off at the start of page 2's dictionary,
    # since each page's content stream follows its own page dictionary.
    stream_cutoff = len(raw)
    if n_pages == 2:
        stream_cutoff = page_starts[1]
        verbosePrint("2-page PDF: parsing only bytes before page 2 (byte %d)\n", stream_cutoff)

    # Extract content stream
    parts = []
    pos = 0
    n_streams = 0

    while pos < stream_cutoff - 6:
        s_idx = raw_str.find("stream", pos)
        if s_idx < 0 or s_idx >= stream_cutoff:
            break

        # skip 'endstream'
        if s_idx >= 3 and raw_str[s_idx - 3:s_idx] == "end":
            pos = s_idx + 6
            continue

        # look back up to 500 bytes for the object dictionary
        hdr = raw_str[max(0, s_idx - 500):s_idx]
        if "/FlateDecode" not in hdr:
            verbosePrint("Skipping non-FlateDecode stream at byte %d\n", s_idx)
            pos = s_idx + 6
            continue

        # the 'stream' keyword is followed by exactly one line ending (\n or \r\n)
        data_start = s_idx + 6
        while data_start < len(raw) and raw[data_start] in (13, 10):
            data_start += 1

        # use the dictionary /Length; searching for 'endstream' can corrupt the compressed binary data
        m = re.search(r"/Length\s+(\d+)(?!\s+\d+\s+R)", hdr)
        if m is None:
            verbosePrint("Skipping stream at byte %d: indirect /Length\n", s_idx)
            pos = s_idx + 6
            continue

        stream_len = int(m.group(1))
        data_end = data_start + stream_len          # exclusive
        if data_end > len(raw):
            verbosePrint("Skipping stream at byte %d: /Length exceeds file size\n", s_idx)
            pos = s_idx + 6
            continue

        decompressed = _inflate_bytes(raw[data_start:data_end])
        if decompressed:
            parts.append(decompressed.decode("latin-1"))
            n_streams += 1

        pos = data_end + 8                          # move past 'endstream'

    content = "".join(parts)
    if not content:
        raise LoadMusePdfError("load_musepdf: No decompressible content streams found.")

    verbosePrint("# streams found: %d \n", n_streams)
    verbosePrint("Decompressed content: %d chars\n", len(content))

    # Calibration signals
    # Verify standard ECG calibration by scanning text in the content stream.  
    # PDF text shown with the Tj operator appears as (string)Tj or [(string)(string)...]TJ.  
    # Just look for the calibration substrings anywhere in the stream.

    # Require that sweep speed is 25 mm/s
    # Program does not support 50 mm/s (which is barely used) or 12.5 mm/s
    # which cannot be used to print a MUSE PDF (although can be used for
    # display on computer)
    if re.search(r"25\s*mm/s", content) is None:
        found_cal = re.findall(r"\d+(?:\.\d+)?\s*mm/s", content)
        raise LoadMusePdfError(
            "load_musepdf: Expected 25 mm/s calibration. Found: %s" % ", ".join(found_cal)
        )

    # Can allow variable gain for voltage, BUT all gains must be the same for
    # the full ECG - function does not support different gains for limb and
    # precordial leads which can theoretically show up.  MUSE indicates
    # different gains for limb and precordial leads by 5,10 mm/s, so to detect
    # an ECG with mixed gains we search for #,# mm/s
    if re.search(r"\d+\s*,\s*\d+\s*mm/mV", content) is not None:
        raise LoadMusePdfError(
            "load_musepdf: PDF has mixed gain for limb and precordial leads. "
            "Only single gain is supported"
        )

    gain_pdf = re.findall(r"(\d+(?:\.\d+)?)\s*mm/mV", content)
    if not gain_pdf:
        raise LoadMusePdfError("load_musepdf: No gain calibration found in PDF content stream")
    mm_per_mv = float(gain_pdf[0])

    m = re.search(r"Td \((\d+)Hz\) Tj ET", content)
    lpf_freq = float(m.group(1)) if m else float("nan")

    #######################################################################################
    # PARSE PDF CONTENT STREAM INTO POLYLINES

    # % PDF path operators:
    # %  x y m       - moveto (start subpath)
    # %  x y l       - lineto (extend subpath)
    # %  S           - stroke and end path

    # % Legacy operators that are not currently used but remain for possible use
    # % in the future
    # %  r g b RG    - set stroke RGB color
    # %  g G         - set stroke gray
    # %  cm          - cm Matrix

    # Split on whitespace into cell array 'tokens'
    tokens = content.split()

    # Number of tokens
    n_tokens = len(tokens)
    verbosePrint("Tokens parsed: %d\n", n_tokens)

    # Classify and parse all numeric tokens up front (MUCH FASTER!)
    nums = np.full(n_tokens, np.nan)
    for i, t in enumerate(tokens):
        c = t[0]
        if ("0" <= c <= "9") or c in "-.+":
            try:
                nums[i] = float(t)
            except ValueError:
                pass                                # stays NaN -> treated as operator

    # Initialize data parser for polyline data extraction
    # All completed polylines so far
    polylines = []     

    # Colors of completed polylines so far (not used anymore)
    pl_colors = []          

    # Polyline being built right now; 2-column matrix that accumulates [x, y] points
    current_pl = []         

    # Current stroke RGB color
    stroke = [0.0, 0.0, 0.0]

    # Stack of pending numeric operands
    # Row vector that grows or shrinks as numbers are pushed or operators consume them
    op_stack = []  

    # cm matrix for conversion between pdf units and physical units       
    cm_matrix = None

    # For each token, if it's a number (nums is not NaN) then it's an operand -> push it into opStack and move on. 
    # If it is not a number (nums is NaN) then it must be an operator.
    for i in range(n_tokens):
        v = nums[i]
        if not math.isnan(v):
            op_stack.append(v)
            continue

        t = tokens[i]

        if t == "m":
            # moveto means start drawing somewhere new:
            # Pull the last two operands from the stack — that's the new (x, y) coordinate.
            
            # Delete any polyline that was already in progress. PDF lets a path contain multiple 
            # disjoint subpaths within a single stroke, but for our purposes each moveto marks 
            # a logical break, and we want each subpath as its own polyline.  If currentPL has 
            # at least 2 points (a real polyline, not just an unfinished move), save it to 
            # polylines along with the current color.
            
            # Start a fresh currentPL containing just this one new point.
            
            # Clear 'opStack'

            if len(op_stack) >= 2:
                # Take most recent (x,y) which is last in opStack
                x, y = op_stack[-2], op_stack[-1]

                if len(current_pl) >= 2:
                    polylines.append(np.array(current_pl, dtype=float))
                    pl_colors.append(list(stroke))
                current_pl = [[x, y]]
            op_stack = []

        elif t == "l":
            # lineto: extend the current subpath:
            # Append a new [x, y] row to the current polyline. This is how the waveforms accumulate --
            # an ECG trace is a moveto followed by 5000 linetos.

            if len(op_stack) >= 2:
                current_pl.append([op_stack[-2], op_stack[-1]])
            op_stack = []

        elif t == "S":
            # stroke: end of path:
            # The polyline is complete, save it, Flush currentPL to polylines, reset to empty, clear the stack.

            if len(current_pl) >= 2:
                polylines.append(np.array(current_pl, dtype=float))
                pl_colors.append(list(stroke))
            current_pl = []
            op_stack = []

        # 'RG' and 'G' set stroke color
        elif t == "RG":
            if len(op_stack) >= 3:
                stroke = [op_stack[-3], op_stack[-2], op_stack[-1]]
            op_stack = []

        elif t == "G":
            # If grayscale, assign same value to RGB

            if len(op_stack) >= 1:
                g = op_stack[-1]
                stroke = [g, g, g]
            op_stack = []

        elif t == "cm":
            #  Get matrix converting units to physical units 
            if len(op_stack) >= 6:
                cm_matrix = list(op_stack[-6:])     # [a b c d e f]
                if (cm_matrix[0] != 0 or cm_matrix[3] != 0
                        or abs(cm_matrix[1]) != abs(cm_matrix[2])):
                    raise LoadMusePdfError(
                        "load_musepdf: Unexpected cm matrix: [%s]. Expected MUSE-style "
                        "rotated transform with a=d=0 and |b|=|c|."
                        % "".join("%g" % x for x in cm_matrix)
                    )
            op_stack = []

        else:
            # Anything else
            # Reset opStack
            op_stack = []

    # For safety, after the loop, check whether currentPL still contains a valid polyline (at least 2 points) and, if so, save it.
    if len(current_pl) >= 2:
        polylines.append(np.array(current_pl, dtype=float))
        pl_colors.append(list(stroke))

    # Confirm we extracted cmMatrix
    if cm_matrix is None or len(cm_matrix) != 6 or any(math.isnan(c) for c in cm_matrix):
        raise LoadMusePdfError("load_musepdf: No valid cm transform found in PDF content stream")

    # Number of polylines
    n_polylines = len(polylines)

    #######################################################################################
    # PULL ECG LEADS and CALIBRATION PULSES OUT OF SET OF POLYLINES

    text_coords = np.full((len(lead_order), 2), np.nan)
    rhythm_text_coords = []
    rhythm_lead = []

    # Verify lead ordering is expected
    for k, lead in enumerate(lead_order):
        pattern = (r"BT\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+Td\s+\("
                   + re.escape(lead) + r"\s*\)\s+Tj\s+ET")
        lead_tok = re.findall(pattern, content)
        if lead_tok:
            text_coords[k, :] = [float(lead_tok[0][0]), float(lead_tok[0][1])]
            if len(lead_tok) == 2:                  # second label = rhythm strip
                rhythm_text_coords.append([float(lead_tok[1][0]), float(lead_tok[1][1])])
                rhythm_lead.append(lead)

    # Sort the textCoords - should be in the order of 1 through 12.  If not
    # then something is non-standard about the PDF and throw error
    sort_idx = _sortrows_asc_desc(text_coords[:, 0], text_coords[:, 1])

    # We use the textCoords to verify the difference between cal signals/baselines
    # as well, as the difference between textCoords should be the same as difference
    # between cal pulses (except for 12x1 where we have to use the textCoords
    # as the way to find where the missing cal pulses would be located.  We
    # take the diff of textCoords and get the diffY that is associated with
    # diffX = 0 (same column)

    diff_tc = np.diff(text_coords, axis=0)
    same_col = np.flatnonzero(diff_tc[:, 0] == 0)
    cal_delta_vals = np.abs(np.unique(diff_tc[same_col, 1]))
    if cal_delta_vals.size != 1:
        raise LoadMusePdfError("load_musepdf: Calibration of lead name locations is not consistent")
    cal_delta = float(cal_delta_vals[0])

    if not np.array_equal(sort_idx, np.arange(12)):
        raise LoadMusePdfError(
            "load_musepdf: Extraction of lead order from PDF text does not match expected lead orders"
        )

    if rhythm_text_coords:
        rtc = np.array(rhythm_text_coords, dtype=float)
        r_sort = _sortrows_asc_desc(rtc[:, 0], rtc[:, 1])
        rhythm_lead = [rhythm_lead[i] for i in r_sort]

    # Find the ECG waveforms (long black polylines)
    # After the parser runs, polylines contains everything the PDF drew with stroked paths — 
    # typically MANY entries: the 12 waveforms we care about, calibration pulses, grid lines, and 
    # other lines. Two characteristics reliably distinguish waveforms from everything else: 
    # they are long (lots of points) and black (or near-black). The Grid is short - only 2 
    # points.  Calibration signals are 60 points

    # nPts is a numeric vector the same length as polylines, with each entry being the point count of the 
    # corresponding polyline. In general there are lots of very short polylines, and then there should 
    # be 12 polylines that are long (>500 pts at 500 Hz) which are the 12 ECG leads
    n_pts = np.array([p.shape[0] for p in polylines], dtype=int)

    #  Each entry in plColors is a RGB triple. Checks whether the maximum of those three components is below 0.3 — 
    # simple "is this color reasonably dark" test so it wont fail if the ECG is not perfect black.  
    # This removes any pink grid lines (not color filtering anymore - just using length)
    is_black = np.array([max(c) < 0.3 for c in pl_colors], dtype=bool)

    # Emperically chose 500 samples as a minimum length as this can be used for 12 lead rhythm strips and 
    # other formats like 4x3 with shorter leads
    # Find indices where polyline is black AND has more than 500 points.
    cand_idx = np.flatnonzero(is_black & (n_pts > 500))
    if cand_idx.size < 12:
        raise LoadMusePdfError(
            "load_musepdf: Expected at least 12 long black polylines corresponding to "
            "lead data but only found %d." % cand_idx.size
        )

    # Keep the 12 polylines of equal length; longer ones are rhythm strips candIdx is the index of all 
    # extracted polylines that are the ECG leads.  This will be 12 for a rhythm strip ECG and 13-15 
    # for a 4x3 ECG with 1-3 rhythm strip leads at the bottom.

    # At this point we have 12-15 waveform polylines but in arbitrary order which is
    # the order they happened to be drawn in the PDF, which isn't necessarily top-to-bottom on the page.
    # The waveforms could be in any sequence (although usually not). We need to sort them by vertical position.

    # For 12 lead rhythm strips:
    # higher stream-y values correspond to higher positions on the displayed page. So lead I (top of the printout)
    # has the highest median y, lead V6 (bottom) has the lowest. Sorting descending puts I first, V6 last

    # However, for the 4x3 leads we also need to sort by horizontal position and vertical position

    polyline_len = n_pts[cand_idx]
    modal_len = _mode(polyline_len)
    cand_idx12 = cand_idx[polyline_len == modal_len]
    rhythm_idx = cand_idx[polyline_len != modal_len]
    num_leads12 = cand_idx12.size
    num_rhythm = cand_idx.size - num_leads12

    if num_leads12 < 12:
        raise LoadMusePdfError(
            "load_musepdf: Expected at least 12 ECG leads of same length but only found %d."
            % num_leads12
        )

    # Polylines come out in drawing order, so sort them into page order:
    # column (x of the first sample) ascending, then row (median y) descending.
    x_baseline = np.array([polylines[i][0, 0] for i in cand_idx12])
    y_baseline = np.array([np.median(polylines[i][:, 1]) for i in cand_idx12])
    cand_idx12 = cand_idx12[_sortrows_asc_desc(x_baseline, y_baseline)]

    #######################################################################################
    # EXTRACT CALIBRATION SIGNALS AND OFFSETS-
    
    #  Count number of calibration pulses which are 60 samples long and get indices
    # Also get baseline values and number of stream units in the cal signal height
    cal_idx = np.flatnonzero(n_pts == 60)
    n_cal = cal_idx.size
    cal_baseline = np.zeros(n_cal)

    # Pull out cal signal polylines and confirm these are cal signals: 
    # should have 2 values with start and end the same. This is also a backup
    # way to catch ECGs with different gains for different sets of leads
    # Also extract the baseline stream values (first sample)
    for k, ci in enumerate(cal_idx):
        cal_sig = polylines[ci]
        cal_baseline[k] = cal_sig[0, 1]
        y = cal_sig[:, 1]
        if np.unique(y).size != 2 or y[0] != y[-1]:
            raise LoadMusePdfError("load_musepdf: Calibration signals are misformed.")

    # Sort the baselines in reverse order to get the baselines in order of top to bottom
    sorted_baseline = np.sort(np.unique(cal_baseline))[::-1]   # top to bottom

    # Validate cal signals and lead names have the same difference between
    # leads as a check for all formats except 12x1.
    if sorted_baseline.size > 1:
        base_diffs = np.abs(np.unique(np.diff(sorted_baseline)))
        if not np.any(base_diffs == cal_delta):
            raise LoadMusePdfError(
                "load_musepdf: Calibration signals and lead name labels are not consistently spaced."
            )
    else:
        # If in 12x1 format we don't have multiuple baselines, so we use the single
        # calibration pulse (for lead I) and then use calDelta to get the rest
        sorted_baseline = sorted_baseline[0] - np.arange(12) * cal_delta


    #######################################################################################
    # EXTRACT GRID VALUES

    # Because the cal pulses and grid are themselves quantized twice, we can't
    # completely rely on single cal pulse to define what 1 mV is in stream
    # units.  As we found, the cal pulse is 1000 stream units at 10 mm/mV, 500
    # stream units at 5 mm/mV, and 2001 stream units at 20 mm/mV.  So rounding
    # can take the real value and push it to a different stream integer.  It is
    # better to use the grid which is many doubly quantized values at a clear
    # interval.  If there is any drift in the number of stream units per mV,
    # this would show up as differences in the number of samples between grid
    # lines in the Y axis.

    # Each Y grid delta = 1 mm

    xmin = np.array([p[:, 0].min() for p in polylines])
    xmax = np.array([p[:, 0].max() for p in polylines])
    ymin = np.array([p[:, 1].min() for p in polylines])
    ymax = np.array([p[:, 1].max() for p in polylines])
    ylen = ymax - ymin

    is_grid = n_pts == 2
    is_horz = is_grid & (ymin == ymax) & (xmax > xmin)          # constant Y

    # Have to filter out the very short vertical lines that mark when a lead
    # changes in a row.  Extract the legnth of the lines not just the points
    # for the time grid.  No such lines exist for the voltage grid so this does
    # not have to be done for voltage
    is_vert = is_grid & (xmin == xmax) & (ymax > ymin) & (ylen > 0.8 * ylen[is_grid].max())

    hy = np.unique([polylines[i][0, 1] for i in np.flatnonzero(is_horz)])
    hx = np.unique([polylines[i][0, 0] for i in np.flatnonzero(is_vert)])

    upm = np.unique(np.diff(hy))
    upms = np.unique(np.diff(hx))

    # Error if grid spacing is not equal
    if upm.size != 1 or upms.size != 1:
        raise LoadMusePdfError("load_musepdf: Grid is misformed.")

    units_per_mm = float(upm[0])                    # 1 mm of grid
    units_per_mm_sec = float(upms[0])

    verbosePrint("Polylines: %d total, %d calibration, %d gridlines, %g unit grid spacing, "
           "%d long, %d long+black (ECG leads)\n",
           n_polylines, n_cal, int(is_grid.sum()), units_per_mm,
           int((n_pts > 1000).sum()), int((is_black & (n_pts > 1000)).sum()))

    #######################################################################################
    # CALCULATE UNITS/SEC AND UNITS/MV

    # Stream units/mV is based on the grid spacing and the mm/mV gain extracted from the PDF. 

    # Calculate units per mV and units per sec from grid and gain
    units_per_mv = units_per_mm * mm_per_mv
    units_per_sec = units_per_mm_sec * 25           # 25 mm/s enforced above

    alpha = units_per_mm * mm_per_mv * (MUV_PER_LSB / 1000.0)

    # Need to get alpha in fractonal form for later certification tests
    # Find fractional representation of muV_per_LSB because this will have
    # fewer decimal places and just makes evrything easier
    s_num, s_den = _rat(MUV_PER_LSB, 1e-12)

    # alpha = units_per_mm * mm_per_mv * (muV_per_LSB/1000), so we have to
    # multiply the numerator by units_per_mm * mm_per_mv and multiply the
    # demoninator by 1000
    numer = _as_int(s_num * units_per_mm * mm_per_mv, "alpha numerator")
    denom = _as_int(s_den * 1000, "alpha denominator")

    # We need the fraction in lowest terms so gcd(ap,aq) = 1
    g_alpha = math.gcd(numer, denom)
    ap = numer // g_alpha
    aq = denom // g_alpha

    if ap <= aq:
        raise LoadMusePdfError("load_musepdf: alpha <= 1, encoding is not reversible")

    verbosePrint("Alpha = %.15g stream/ADU\n", alpha)
    verbosePrint("Alpha = %d/%d stream/ADU in fractional form\n", ap, aq)


    # Calculate sampling frequency fs using the first ECG lead
    stream_x = polylines[cand_idx12[0]][:, 0]

    # Set time to start at t=0 by subtracting the first streamX
    t = (stream_x - stream_x.min()) / units_per_sec

    # Make sure uniform samples and calculate the sampling frequency to use later
    dt = np.diff(t)

    # Each delta time is not perfectly exact due to this floating point tolerance
    # The tolerance is on the order of 1x-12, but we will impose a tighter
    # tolerance just to be sure there are no issues
    tol = 1.0 / units_per_sec                       # 1 stream unit = 0.4 ms for MUSE
    if np.max(np.abs(dt - np.median(dt))) > tol:
        raise LoadMusePdfError(
            "load_musepdf: Time samples are not uniformly spaced and exceed "
            "1 stream unit of jitter"
        )
    
    # Calculate the sampling freq
    # round just in case some sneaky floating point issue returns
    fs = int(round((len(t) - 1) / (t[-1] - t[0])))


    #######################################################################################
    # DETERMINE ECG LAYOUT

    #Shorten variable name
    SB = sorted_baseline

    #  Determine ECG layout here so we can use the correct lead offsets for the specific ECG layout
    # offsets variable includes the offset values in ADU units for the 12 main leads in standard order
    lead_len = int(n_pts[cand_idx12[0]])

    # Signal length should be fs * duration in sec
    # 12L rhythm strip should be 10 sec.  Set at 9 sec in case some missing
    # samples or issues with rounding freq etc
    if lead_len > fs * 9:
        ecg_format_string = "12 rhythm strips"
        ecg_format_id = "12LRhythm"
        offsets = SB
        rhythm_offsets = np.array([])

    # 6x2 5 sec format has no rhythm strips but due to lead switching each non rhythm 
    # strip lead is 12 samples shorter than would be expected   So the length will be longer 
    # than 2.5 sec but shorter than 5 sec.  This format never has a rhythm strip.
    elif fs * 2.5 < lead_len <= fs * 5:
        ecg_format_string = "6x2 5s and %d rhythm strips" % num_rhythm
        ecg_format_id = "6x2"
        offsets = np.tile(SB[:6], 2)
        rhythm_offsets = np.array([])

    else:
        # Everything else should be 4x3 2.5 sec with 0 or 3 rhythm strips
        if num_rhythm not in (0, 1, 3):
            raise LoadMusePdfError(
                "load_musepdf: Number of rhythm strips (%d) is not supported" % num_rhythm
            )
        ecg_format_string = "4x3 2.5s and %d rhythm strips" % num_rhythm
        ecg_format_id = "4x3+%d" % num_rhythm
        offsets = np.tile(SB[:3], 4)
        rhythm_offsets = SB[3:3 + num_rhythm] if num_rhythm else np.array([])

    verbosePrint("PDF data format: %s\n", ecg_format_string)

    #######################################################################################
    # CONVERT SIGNALS TO ADU UNITS

    # Structure for storing extracted signals
    signals = {}
    clip_data = {}
    diagnostics = {k: {} for k in
                   ("certified", "W", "WMargin", "passWLimit", "passW", "GCD",
                    "numD", "passGCD", "BOut", "BOutClasses", "passB")}
    total_clipping = 0
    hi_clip = np.zeros(12)
    lo_clip = np.zeros(12)

    # Convert the 12L polylines into numeric signals (this does not include
    # rhythm strips on 4x3 format ECGs

    for k in range(12):
        stream_y = polylines[cand_idx12[k]][:, 1]
        lead = lead_order[k]

        # Clipping detection on raw stream based on max (21150) and min (450) stream values using min of 1 sample
        # If any lead (although its usually lead I) equal to or exceeds 21150 units the lead is clipped at this value
        # If any lead (although its usually lead V6) is equal to or below 450 units the lead is clipped at this value
        # We use 1 sample as indicating clipping because we can't tell if the 1 sample just reached max/min or if it 
        # was signifcantly larger, and even single point peak clipping can introduce errors. Will let user decide how 
        # to proceed if clipping is detected, as ANY clipping can reduce accuracy of the signal extraction.
    
        clip_mask = _detect_clipping(stream_y, 1)

        # Now use double quantization features to recover ADU units exactly.
        adu, d = _stream2adu(stream_y, alpha, ap, aq, offsets, k)

        # Assign signal to structure
        signals[lead] = adu

        # Assign diagnostics to structure for debug purposes
        diagnostics["certified"][lead] = d["certified"]
        diagnostics["W"][lead] = d["W"]
        diagnostics["WMargin"][lead] = d["Wmargin"]
        diagnostics["passWLimit"][lead] = d["pass_W_limit"]
        diagnostics["passW"][lead] = d["pass_W"]
        diagnostics["GCD"][lead] = d["gcd"]
        diagnostics["numD"][lead] = d["numD"]
        diagnostics["passGCD"][lead] = d["pass_gcd"]
        diagnostics["BOut"][lead] = d["B_out"]
        diagnostics["BOutClasses"][lead] = d["B_out_classes"]
        diagnostics["passB"][lead] = d["pass_B"]

        # Save clipping data into structure
        clip_samples = np.flatnonzero(clip_mask)
        clip_data[lead] = int(clip_mask.sum())
        clip_data[lead + "_clipsamples"] = clip_samples
        total_clipping += clip_samples.size

        hi_clip[k] = stream_y.max()
        lo_clip[k] = stream_y.min()

        if verbose:
            print("Lead %s: %d samples" % (lead, adu.size))
            if d["certified"] == 0:
                print("   Lead %s Certification Failed!" % lead)
            if _isnan(d["certified"]):
                print("   Lead %s Certification Indeterminant!" % lead)
            if clip_mask.any():
                print("   Lead %s: %d clipped samples (%.2f%%); data clipped to "
                      "range ~[%d, %d] units"
                      % (lead, clip_mask.sum(), 100 * clip_mask.sum() / clip_mask.size,
                         lo_clip[k], hi_clip[k]))


    #######################################################################################
    # CALCULATED LEADS
    # 'pdf'  :: keep all PDF-extracted values
    # 'calc' :: reconstruct leads III, aVR, aVL, aVF 

    leads_to_calc = [] if leads_mode == "pdf" else ["III", "aVR", "aVL", "aVF"]

    # If some leads need to be calculated
    if leads_to_calc:
        formulas = {
            "III": lambda s: -s["I"] + s["II"],
            "aVF": lambda s: s["II"] - 0.5 * s["I"],
            "aVR": lambda s: -0.5 * s["I"] - 0.5 * s["II"],
            "aVL": lambda s: s["I"] - 0.5 * s["II"],
        }

        # Any reconstructed lead inherits clipped samples from its two source leads
        shared_clip = np.unique(np.concatenate(
            [clip_data["I_clipsamples"], clip_data["II_clipsamples"]]
        )).astype(int)

        for ld in leads_to_calc:
            signals[ld] = formulas[ld](signals)
            clip_data[ld + "_clipsamples"] = shared_clip
            clip_data[ld] = shared_clip.size

        total_clipping = sum(clip_data[ld + "_clipsamples"].size for ld in lead_order)

        # Also have to redo the certification of the calculated leads, because
        # a lead could clip/fail certification using the waveforms extracted
        # from the PDF, but then be "rescued" by using lead I and II which are
        # not clipped and pass certification.  We can't just pass the
        # calculated leads through the certification tests because aVR, aVL,
        # and aVF do not exist in the original stream units (due to division by 2), 
        # and the tests therefore do not apply.  We will have the calculated
        # leads inheret the certification of leads I and II. If leads I and II
        # are certified bit exact then any derivation from them is also
        # guarenteed to be bit exact.  If lead I or II fails certification, then the
        # calculated leads fail certificatoin.  If Lead I or II is indeterminant (NaN), 
        # then the calculated leads are also indeterminant (NaN).  We also do
        # not pass along the numeric values of W etc because they may not apply

        certs = [diagnostics["certified"]["I"], diagnostics["certified"]["II"]]
        if any((not _isnan(c)) and c == 0 for c in certs):
            calc_cert = 0
        elif any(_isnan(c) for c in certs):
            calc_cert = float("nan")
        else:
            calc_cert = 1

        original = {k: {} for k in diagnostics}
        for ld in leads_to_calc:
            for key in diagnostics:
                original[key][ld] = diagnostics[key][ld]
                # numeric diagnostics don't carry over to derived leads
                diagnostics[key][ld] = float("nan")
            diagnostics["certified"][ld] = calc_cert

            if verbose:
                print("   Calculated Lead %s Certification %s"
                      % (ld, "Passed" if calc_cert == 1 else "Failed!"))

        diagnostics["calcleadsOriginal"] = original

    lead_lengths = [signals[ld].size for ld in lead_order]
    if len(set(lead_lengths)) != 1:
        raise LoadMusePdfError(
            "load_musepdf: Inconsistent lead lengths: [%s]"
            % " ".join(str(n) for n in lead_lengths)
        )

    # Now deal with Rhythm strips (4x3 formats only)
    total_clipping_rhythm = 0
    if num_rhythm > 0:
        signals["rhythm"] = {}
        clip_data["rhythm"] = {}
        diagnostics["rhythm"] = {k: {} for k in
                                 ("certified", "W", "WMargin", "passWLimit", "passW",
                                  "GCD", "numD", "passGCD", "BOut", "BOutClasses",
                                  "passB")}

        for k in range(num_rhythm):
            R = polylines[rhythm_idx[k]][:, 1]
            lead = rhythm_lead[k]

            # Clipping detection on raw stream based on max (21150) and min (450) stream values using min of 2 samples
            # If any lead (although its usually lead I) exceeds 21150 units the lead is clipped at this value
            # If any lead (although its usually lead V6) is below 450 units the lead is clipped at this value
            clip_mask = _detect_clipping(R, 1)

            # R currently is in stream units, so will convert to adu and subtract the position offset to zero signal as 
            # is done for the individual leads
            adu, d = _stream2adu(R, alpha, ap, aq, rhythm_offsets, k)

            # Write output to signals.rhythm while we are here
            # signals.rhythm does not show up for 12 lead rhythm strips (if numRhythm = 0)
            signals["rhythm"][lead] = adu

            # Assign diagnostics to structure for debug purposes
            rd = diagnostics["rhythm"]
            rd["certified"][lead] = d["certified"]
            rd["W"][lead] = d["W"]
            rd["WMargin"][lead] = d["Wmargin"]
            rd["passWLimit"][lead] = d["pass_W_limit"]
            rd["passW"][lead] = d["pass_W"]
            rd["GCD"][lead] = d["gcd"]
            rd["numD"][lead] = d["numD"]
            rd["passGCD"][lead] = d["pass_gcd"]
            rd["BOut"][lead] = d["B_out"]
            rd["BOutClasses"][lead] = d["B_out_classes"]
            rd["passB"][lead] = d["pass_B"]

            # Save clipping data into structure
            clip_samples = np.flatnonzero(clip_mask)
            clip_data["rhythm"][lead] = int(clip_mask.sum())
            clip_data["rhythm"][lead + "_clipsamples"] = clip_samples
            total_clipping_rhythm += clip_samples.size

            if verbose:
                print("Rhythm strip %s: %d samples" % (lead, adu.size))
                if d["certified"] == 0:
                    print("   Rhythm Strip %s Certification Failed!" % lead)
                if _isnan(d["certified"]):
                    print("   Rhythm Strip %s Certification Indeterminant!" % lead)
                if clip_mask.any():
                    print("   Rhythm Strip %s: %d clipped samples (%.2f%%); Data clipped "
                          "to range ~[%d, %d] units"
                          % (lead, clip_mask.sum(), 100 * clip_mask.sum() / clip_mask.size,
                             R.min(), R.max()))

    #######################################################################################
    # CONVERT UNITS IF NEEDED
    # If outputting to mV (nominal) now convert ADU units to mV (1 ADU = 4.88 microvolts)

    if not outputadu:
        # Convert ADU to mV
        signals["units"] = "mV"
        for lead in lead_order:
            signals[lead] = signals[lead] * (MUV_PER_LSB / 1000.0)
        if num_rhythm > 0:
            for lead in rhythm_lead:
                signals["rhythm"][lead] = signals["rhythm"][lead] * (MUV_PER_LSB / 1000.0)
    else:
        signals["units"] = "ADU"

    # Export total number of clipped points - put in signals and info
    clip_data["totalClipping"] = total_clipping
    signals["totalClipping"] = total_clipping
    if num_rhythm > 0:
        clip_data["totalClippingRhythm"] = total_clipping_rhythm
        signals["totalClippingRhythm"] = total_clipping_rhythm

    signals["hz"] = fs
    verbosePrint("Sampling frequency = %d Hz\n", fs)

    #######################################################################################
    # OVERALL CERTIFICATION

    # Make a single flag for if any of the validation test failed
    # If any lead certification is 0 then the ECG fails

    ecg_certified = 1
    for v in diagnostics["certified"].values():
        if (not _isnan(v)) and v == 0:
            ecg_certified = 0
            break

    if "rhythm" in diagnostics and ecg_certified == 1:
        for v in diagnostics["rhythm"]["certified"].values():
            if (not _isnan(v)) and v == 0:
                ecg_certified = 0
                break

    if ecg_certified == 1:
        if any(_isnan(v) for v in diagnostics["certified"].values()):
            ecg_certified = float("nan")
        elif "rhythm" in diagnostics and any(
                _isnan(v) for v in diagnostics["rhythm"]["certified"].values()):
            ecg_certified = float("nan")

    if verbose:
        if ecg_certified == 1:
            print("\n*ECG Certification Passed!*")
        elif ecg_certified == 0:
            print("\n*ECG Certification Failed!*")
        else:
            print("\n*ECG Certification Indeterminant!*")

    #######################################################################################
    # WRITE EXTRA DATA TO INFO

    info = {
        "hz": fs,
        "leadOrder": list(lead_order),
        "leadsMode": leads_mode,
        "ecgFormat": ecg_format_id,
        "ecgFormatStr": ecg_format_string,
        "numPolylines": n_polylines,
        "numLeadsDetected": int((is_black & (n_pts > 1000)).sum()),
        "cmMatrix": np.array(cm_matrix, dtype=float),
        "cm2": cm_matrix[1],
        "certified": ecg_certified,
        "diagnostics": diagnostics,
        "XMLmicrovoltsLSB": MUV_PER_LSB,
        "lowPassFreq": lpf_freq,
        "unitsPerSec": units_per_sec,
        "unitsPerMv": units_per_mv,
        "mmPerMv": mm_per_mv,
        "mmPerSec": 25,
        "alpha": alpha,
        "alphaFraction": (ap, aq),
        "offsets": SB,
        "numPages": n_pages,
        "numStreams": n_streams,
        "clipping": clip_data,
        "maxStreamUnits": hi_clip,
        "minStreamUnits": lo_clip,
    }

    # Add rhythm strip info for 4x3 format
    if num_rhythm > 0:
        info["rhythmStripLeads"] = list(rhythm_lead)

    # Add certification to signals structure too
    signals["certified"] = ecg_certified

    # Export as csv
    if exportfile:
        export_name = _pdfsig2csv(signals, pdf_file, outputadu)
        verbosePrint("\nExported %s to %s\n\n", pdf_file, export_name)

    return signals, info


#######################################################################################
# COMMAND LINE

if __name__ == "__main__":
    import argparse

    ap_ = argparse.ArgumentParser(description="Digitize a GE MUSE 12-lead ECG PDF.")
    ap_.add_argument("pdf")
    ap_.add_argument("options", nargs="*", help="any of: " + ", ".join(sorted(VALID_OPTIONS)))
    args = ap_.parse_args()

    sig, nfo = load_musepdf(args.pdf, *args.options)
    print("Format: %s | %d Hz | certified: %s"
          % (nfo["ecgFormatStr"], nfo["hz"], nfo["certified"]))