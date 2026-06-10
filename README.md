# PaddleOCR HTR Pipeline for Ancient Chinese Manuscripts

Scripts for driving PaddleOCR to generate base segmentation of vertically-written ancient Chinese manuscripts, producing PAGE-XML output compatible with [eScriptorium](https://escriptorium.fr/).

Developed as part of a handwritten text recognition (HTR) workflow for historical Chinese and Sino-Japanese manuscripts held in the Penn Libraries digital collections.

---

## Background and motivation

eScriptorium's built-in segmentation performs poorly on vertically-written ancient Chinese. This pipeline uses PaddleOCR's detection engine as a first pass to generate bounding boxes and baselines in PAGE-XML format, which can then be imported into eScriptorium for manual correction and ground truth production.

The scripts evolved over several months of working with two manuscripts: *Yōso zusetsu* (廱疽圖說), an illustrated Sino-Japanese medical treatise with irregular handwriting, and *Xing li da quan shu* (性理大全書), a densely-typeset Neo-Confucian compilation. The key challenges encountered were:

- PaddleOCR API breaking changes across 3.0.0 → 3.2.x (parameter names, output structure, `ocr()` vs `predict()`)
- eScriptorium requiring a `Baseline` element inside each `TextLine` — missing this causes silent import failure
- Dense vertical layouts where text columns are detected as horizontal text (see the case study section below)
- Pages with show-through, illustrations, or very low contrast requiring preprocessing before detection

The `experimental/` folder preserves the scripts written during debugging. They are not recommended for use but document what was tried and why it was abandoned.

## Test materials

Scripts were developed and tested on two manuscripts from the Penn Libraries digital collections (Colenda Digital Repository), both openly accessible:

- *Yōso zusetsu* (廱疽圖說): Sino-Japanese illustrated medical treatise, irregular handwriting
  https://colenda.library.upenn.edu/catalog/81431-p3806r

- *Xing li da quan shu* (性理大全書): Neo-Confucian compilation, dense vertical typeset layout
  https://colenda.library.upenn.edu/catalog/81431-p39k46864

---

## Quick start

If you are using **PaddleOCR 3.0.0**, start with `production/paddle_single_with_baseline.py` or `production/paddle_batch_v1.py`.

If you are using **PaddleOCR 3.2.x or later**, start with `updated/paddle_batch_v4_vertical_filter.py`.

If your manuscript has a **dense vertical layout** (closely packed columns, small-character commentary, 雙行夾註), use `updated/paddle_batch_v5_native_res.py` and read the case study section below first.

```bash
# Single image, segmentation only (3.0.x)
python3 production/paddle_single_with_baseline.py \
    --image your_page.jpg --outdir out --lang ch --to_pagexml

# Batch processing a folder (3.2.x)
python3 updated/paddle_batch_v4_vertical_filter.py \
    --input_dir images/ --outdir out --lang ch --to_pagexml

# Dense vertical layouts (see case study section)
python3 updated/paddle_batch_v5_native_res.py \
    --image your_page.jpg --outdir out --strategy rotate90 --to_pagexml
```

Then import the `.xml` output into eScriptorium: **Images → Import → Transcription (XML)**.

---

## Repository structure

```
paddleocr-htr-ancient-chinese/
├── README.md
├── experimental/          # Debugging history — not for production use
├── production/            # Stable scripts for PaddleOCR 3.0.x
│   ├── paddle_single_with_baseline.py
│   ├── paddle_single_sauvola.py
│   ├── paddle_single_selective_clahe.py
│   └── paddle_batch_v1.py
├── updated/               # Scripts for PaddleOCR 3.2.x and later
│   ├── paddle_batch_v2_predict_api.py
│   ├── paddle_batch_v3_no_orientation.py
│   ├── paddle_batch_v4_vertical_filter.py
│   └── paddle_batch_v5_native_res.py      # dense vertical layouts — see case study
└── utils/
    ├── util_diagnose_paddle_api.py
    ├── util_merge_page_xml.py
    └── util_compare_det_strategies.py     # strategy benchmark — see case study
```

---

## Script reference

### production/

**`paddle_single_with_baseline.py`**
The first script that produces PAGE-XML accepted by eScriptorium. Key feature: each `TextLine` element includes a `Baseline` computed as the vertical midline between the top and bottom edges of the bounding box — this is required for vertical text in eScriptorium and was absent in all earlier versions. Uses `ocr()` API with `det_db_*` parameters. PaddleOCR 3.0.x.

**`paddle_single_sauvola.py`**
Adds Sauvola binarization preprocessing (`--preprocess binarize_sauvola`). Use this for pages with uneven lighting, show-through from the verso, or low-contrast ink where standard detection misses lines. The Sauvola algorithm (from `skimage`) is better suited to historical documents than Otsu or adaptive thresholding because it accounts for local intensity variation. PaddleOCR 3.0.x.

**`paddle_single_selective_clahe.py`**
Adds CLAHE contrast enhancement with a selective mode that applies enhancement only to low-contrast regions of the image (using a local standard deviation filter), leaving high-contrast regions untouched. Useful when a page has mixed contrast. Heavy time cost. PaddleOCR 3.0.x.

**`paddle_batch_v1.py`**
Batch-processing version of `paddle_single_with_baseline.py`. Takes `--input_dir` to process a whole folder at once. The OCR instance is initialized once and reused across all images. PaddleOCR 3.0.x.

### updated/

The v2 to v4 scripts replace `ocr()` with `predict()` and parse the `OCRResult.json['res']` output structure introduced in PaddleOCR 3.2.x. They also use the renamed initialization parameters (`text_det_*` instead of `det_db_*`).

**`paddle_batch_v2_predict_api.py`**
First working batch script for PaddleOCR 3.2.x. Uses `use_textline_orientation=True`.

**`paddle_batch_v3_no_orientation.py`**
Same as v2 but with `use_textline_orientation=False`. In practice, enabling orientation classification on ancient vertical Chinese worsened results — the classifier was trained on modern text and misread classical column layouts. This is the safer default.

**`paddle_batch_v4_vertical_filter.py`**
Adds a `filter_vertical_boxes()` post-processing step that discards detected boxes with height-to-width ratio below 1.5, and lowers detection thresholds for denser layouts. Works well on regular vertical layouts; insufficient for dense small-character pages (see case study). PaddleOCR 3.2.x+.

**`paddle_batch_v5_native_res.py`**
Adds native-resolution detection, a rotate-90 detection strategy, run modes (`seg` / `seg_rec` / `rec`), and column-aware reading order. Written for the dense-layout problem in *Xing li da quan shu*; full documentation in the case study section below. Validated under both the original pinned 3.0.x environment and 3.6.x.

### utils/

**`util_diagnose_paddle_api.py`**
Run this first if you are getting unexpected errors. It prints the installed PaddleOCR version, confirms the `PaddleOCR` class initializes correctly, and lists the parameters accepted by `ocr()`. Useful for confirming whether you are on a version that uses `ocr()` or `predict()`.

**`util_merge_page_xml.py`**
Merges multiple PAGE-XML files from a folder into a single XML file. Useful for consolidating segmentation results before import.

**`util_compare_det_strategies.py`**
Runs one page through three detection configurations (the v4-equivalent settings, native resolution, and rotate-90) and writes overlay previews plus a stats table. This is the validation instrument used in the case study below; run it on a few representative pages of any new manuscript before choosing a strategy.

### experimental/

These scripts are preserved for reference only. None of them should be used for production work. The single most important lesson from this entire experimental phase: **eScriptorium silently discards `TextLine` elements that do not contain a `Baseline` child element.** No error is raised on import. The segmentation appears to load, but the panel remains blank.

| Script | What it tried | Why superseded |
|--------|--------------|----------------|
| `exp_tesseract_baseline.py` | Tesseract as an alternative to PaddleOCR | Poor performance on vertical ancient Chinese |
| `exp_paddle_v2v7_compat.py` | PaddleOCR 2.7 with partial 3.x compatibility shims | Unreliable on 3.x; superseded by v3_initial |
| `exp_paddle_v3_initial.py` | First attempt at PaddleOCR 3.x | No Baseline in XML; eScriptorium rejected it |
| `exp_paddle_v3_tuned_params.py` | Tuned `det_db_*` parameters, max_side 1200 | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_clahe_global.py` | Global CLAHE contrast enhancement | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_clahe_selective.py` | Selective CLAHE (low-contrast regions only) | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_binarize_v1.py` | Adaptive, Otsu, and Sauvola binarization; box filtering | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_binarize_v2.py` | Refined binarization; fallback on zero detections | No Baseline; eScriptorium rejected it |

---

## Case study: dense vertical layouts in *Xing li da quan shu* (the v5 script)

The v2 to v4 scripts remain fully functional for the layouts they were built on. What follows concerns a specific manuscript property — densely packed vertical columns, especially small-character commentary — that defeats the standard detection configuration regardless of script version.

### The problem

On pages of *Xing li da quan shu* dominated by small-character text (commentary in 雙行夾註 double-row interlinear format, or full pages of small-character annotation), the detector returns wide horizontal boxes spanning many columns instead of one box per column. The columns themselves are perfectly legible to a human reader and the model recognizes individual characters without difficulty; the failure is purely in line formation.

### What v5 adds

`updated/paddle_batch_v5_native_res.py` provides:

- **Native-resolution detection.** Earlier scripts pre-shrank every page to 1200 px before detection; v5 removes this and lets the pipeline run at native resolution (built-in cap 4000 px). An emergency `--pre_max_side` flag remains for memory-limited machines.
- **Three detection strategies.** `native` (direct detection), `rotate90` (rotate the page 90 degrees so columns become horizontal lines, detect, then map all coordinates exactly back to the original image; the mapping is unit-tested as an exact geometric inverse), and `auto` (run native, fall back to rotate90 if the proportion of vertical boxes is low, keep the better result).
- **Three run modes.** `seg` (detection model only), `seg_rec` (detection plus recognition with text prefilled into the XML), and `rec` (read PAGE-XML corrected in eScriptorium, crop each line, recognize, write `TextEquiv` back — closing the loop of the eScriptorium workflow).
- **Column-aware reading order.** Lines are clustered into columns by x-center before sorting right-to-left, then top-to-bottom within each column.
- Calmer detection defaults for dense layouts: `unclip_ratio=1.3` (v4 used 2.0, which dilates boxes into neighboring columns), `box_thresh=0.45` (v4 used 0.2, which admits ruling lines and noise).

### Quick start for v5

```bash
# Dense small-character pages: rotate90 is the validated choice
python3 updated/paddle_batch_v5_native_res.py \
    --input_dir images/ --outdir out --strategy rotate90 --to_pagexml

# Irregular pages (prefaces, raised honorifics, mixed sizes): native
python3 updated/paddle_batch_v5_native_res.py \
    --image preface.jpg --outdir out --strategy native --to_pagexml

# Segmentation + prefilled recognition
python3 updated/paddle_batch_v5_native_res.py \
    --input_dir images/ --outdir out --mode seg_rec --strategy rotate90 --to_pagexml

# Recognition only, backfilling eScriptorium-corrected XML
python3 updated/paddle_batch_v5_native_res.py \
    --input_dir images/ --xml_dir corrected_xml/ --outdir out --mode rec
```

All flags are documented in the script header. Faint or show-through pages may need `--box_thresh 0.3` in rotate90 mode (see validation notes on page 4 below).

### Validation results (five representative pages)

Five pages spanning the layout types of *Xing li da quan shu* were run through `util_compare_det_strategies.py`, comparing the v4-equivalent configuration (`legacy1200`), native-resolution detection (`native`), and `rotate90`. The vertical ratio is the share of detected boxes with height-to-width ratio above 1.5.

| Page | Layout | Strategy | Boxes | Vertical | V-ratio | Median width (px) | Visual judgment |
|------|--------|----------|-------|----------|---------|-------------------|-----------------|
| 1 | Preface: mixed large/small scripts, irregular column starts, raised honorifics | legacy1200 | 29 | 9 | 0.31 | 181 | boxes too wide |
| | | native | 28 | 9 | 0.32 | 161 | **best of three; dense area still merges** |
| | | rotate90 | 41 | 14 | 0.34 | 65 | fragmented, overlapping boxes |
| 2 | Name registry: two vertical registers, short columns with small-character notes | legacy1200 | 33 | 26 | 0.79 | 96 | tilting boxes |
| | | native | 32 | 27 | 0.84 | 74 | some boxes too wide, tilting |
| | | rotate90 | 27 | 21 | 0.78 | 84 | **cleanest column capture** |
| 3 | Mixed: large main text plus dense small-character commentary block | legacy1200 | 27 | 5 | 0.19 | 538 | giant undifferentiated box at top |
| | | native | 26 | 5 | 0.19 | 522 | horizontal banding over commentary |
| | | rotate90 | 23 | 16 | 0.70 | 42 | **only strategy resolving the dense block; some characters missed** |
| 4 | Full page of dense small-character text | legacy1200 | 18 | 0 | 0.00 | 763 | pure horizontal stripes |
| | | native | 18 | 0 | 0.00 | 748 | pure horizontal stripes |
| | | rotate90 | 10 | 9 | 0.90 | 56 | **vertical columns; several columns missed** |
| 5 | Mixed scripts, large/small switched sides relative to page 1 | legacy1200 | 29 | 10 | 0.35 | 133 | most chaotic |
| | | native | 25 | 7 | 0.28 | 306 | wide merged boxes |
| | | rotate90 | 17 | 17 | 1.00 | 54 | **no omissions, fully vertical** |

Two findings stand out. First, on dense small-character pages (3, 4, 5), `rotate90` is the only strategy that produces vertical columns at all; `legacy1200` and `native` return horizontal stripes spanning the full text block (median box widths of 500 to 760 px are essentially page-wide). Second, **native resolution alone barely moves the needle**: on pages 1, 3, and 4 the native results are nearly identical to legacy1200, and on page 5 marginally worse. The resolution-starvation hypothesis that motivated v5's design is therefore at most a secondary factor. The decisive intervention is rotation.

### Why this happens: a diagnosis

PaddleOCR's detectors (the DB family) are segmentation-based: they predict a per-pixel text probability map, shrink each text instance to a kernel during training to keep neighbors apart, and re-expand detected kernels with an unclip ratio at inference. A known limitation of this family, stated in the PSENet paper that the approach descends from, is that segmentation-based detectors "may not separate the text instances that are very close to each other" (Wang et al. 2019). Yang et al. (2018), who built the TKH/MTH datasets precisely because of this problem, put the document-side version plainly: "Characters in historical documents are typically densely distributed" and are difficult to localize with standard detectors.

But proximity alone does not explain the validation results, and this is where the data forced a revision. If line formation were a neutral, isotropic function of pixel proximity, then raising the resolution (native) should have separated the columns, and rotating the page should have changed nothing, since rotation preserves all relative distances. The opposite happened. What the horizontal stripes on page 4 reveal is that the detector does not group text isotropically: it has a strong learned prior for **long horizontal line shapes**, acquired from training data in which horizontal lines overwhelmingly dominate. In a dense block of small characters, where the horizontal gap between columns is comparable to or smaller than the vertical gap between characters within a column, that prior wins, and the model confidently reconstructs the layout as horizontal rows — one clean stripe per character row, exactly what the page 4 overlays show. Rotating the page 90 degrees does not change the geometry; it changes which axis the prior is applied to. The true columns become horizontal lines, the modal case of the training distribution, and detection succeeds.

This account also explains where rotate90 fails. Columns that are internally sparse or irregular — page 1's preface, with raised honorifics (抬頭), staggered column starts, and blank gaps inside columns — become, after rotation, short horizontal fragments at uneven offsets. The detector splits them at the gaps, and the fragments map back as overlapping boxes. Rotation helps when columns are internally continuous and densely stacked, and hurts when they are not. The missed columns in pages 3 and 4 under rotate90 are a separate, milder issue: faint ink and show-through depress the mean probability score of a long line, and the default `box_thresh=0.45` prunes it; lowering the threshold for faded pages recovers them.

To our knowledge there is no published study that documents this rotation remedy, or the directional-prior account behind it, for DB-family detectors on vertical Chinese specifically; the explanation above is an inference from the detector's mechanics and from these validation results, consistent with the cited literature but not itself established in it. The closest published discussions are the dense-text detection literature (Yang et al. 2018; Wang et al. 2019) and recent VLM benchmarks on Chinese ancient documents, which independently identify vertical typesetting, interlinear notes, and small characters as the defining challenges of this material (Lin et al., AncientDoc; the CHURRO report likewise attributes zero-shot VLM failures on classical Chinese to models imposing modern horizontal reading-order assumptions on vertical layouts).

### Strategy selection guide

| Page type | Recommended strategy |
|-----------|---------------------|
| Dense small-character pages, 雙行夾註 commentary, internally continuous columns | `rotate90` |
| Prefaces, irregular column starts, raised honorifics, large intra-column gaps | `native` |
| Faded or show-through dense pages | `rotate90 --box_thresh 0.3` |
| Unknown / mixed batch | `auto`, then inspect overlays |

A caution on `auto`: its switching heuristic (vertical-box counts) prefers whichever strategy yields more vertical boxes, which on irregular preface pages can favor rotate90's fragmented output over native's smaller but cleaner result. Page 1 of the validation set is exactly such a case. For batches with known page types, set the strategy explicitly; reserve `auto` for triage.

The deeper limitation, shared by all strategies, is that pages mixing large main text with dense commentary blocks (page 3) have no single best global setting. The principled fix is layout-first processing — detect text regions, classify them by density, and run the appropriate strategy per region — which is left as the natural next step.

### Two manuscripts compared: why irregular handwriting segmented more easily than clean dense print

A counterintuitive observation from this project: *Yōso zusetsu*, with its tilted columns, irregular handwriting, reproduction artifacts, and interleaved illustrations, was consistently easier to keep vertical in base segmentation than the cleanly carved, perfectly regular *Xing li da quan shu*.

The diagnosis above explains why. The difficulty of vertical base segmentation for this detector family is governed not by script neatness but by layout geometry — specifically, the relation between inter-column gaps and intra-column character spacing, and whether that relation lets the detector's horizontal-line prior misfire. *Yōso zusetsu* has few columns per page, generously spaced, with handwriting whose brush flow keeps each column visually continuous from top to bottom; tilt and irregularity are no obstacle, because segmentation-based detectors handle arbitrary shapes well (that is what they were designed for). *Xing li da quan shu*'s small-character commentary inverts the geometry: columns jammed against each other, characters within a column separated by regular gaps, and the 雙行夾註 format actively pairing characters horizontally inside a single column track. Visual noise is a recognition problem; gap geometry is a detection problem. The two manuscripts fail on different axes, and base segmentation lives on the second.

One practical corollary: a manuscript's suitability for this pipeline cannot be read off its apparent cleanliness. Run `util_compare_det_strategies.py` on a few pages first.

---

## Dependencies

The dependency story is layered by script generation. The v5 script was written against the PaddleOCR 3.6.0 source but uses only module APIs (`TextDetection`, `TextRecognition`, `PaddleOCR`) and parameter names that are stable across 3.0.x to 3.6.x; it has been validated under both the original pinned environment below and a current 3.6.x install.

| Scripts | PaddleOCR | Notes |
|---------|-----------|-------|
| `production/` | 3.0.x | `ocr()` API, `det_db_*` parameter names |
| `updated/` v2 to v4 | 3.2.x+ | `predict()` API, `text_det_*` parameter names |
| `updated/` v5, `utils/util_compare_det_strategies.py` | 3.0.x to 3.6.x | module APIs; validated under both |

**Original pinned environment (tested combination, still works for everything in this repo):**
```bash
python3 -m venv ~/venvs/paddleocr-env
source ~/venvs/paddleocr-env/bin/activate
pip install numpy==1.26.4
pip install opencv-python==4.5.5.64
pip install Pillow==9.5.0
pip install paddlepaddle==3.0.0
pip install paddleocr==3.0.0
pip install scikit-image scipy lxml pyyaml
```

**Current environment (if starting fresh):**
```bash
python3 -m venv ~/venvs/paddleocr36-env
source ~/venvs/paddleocr36-env/bin/activate
pip install paddlepaddle paddleocr
```
The `numpy<2.0` pin is required for the 3.0.x stack; PaddleOCR 3.6.x resolved those incompatibilities.

Additional for `paddle_single_sauvola.py` and `selective_clahe.py`: `scikit-image`, `scipy`.

Model weights download automatically on first run. If the default source is unreachable from your network, set `PADDLE_PDX_MODEL_SOURCE=HUGGINGFACE`.

PaddleOCR updates frequently and often introduces breaking changes without deprecation warnings. If you encounter unexpected errors, run `util_diagnose_paddle_api.py` first to confirm your environment.

---

## Known issues and troubleshooting

**eScriptorium import appears to succeed but the segmentation panel is blank.**
The PAGE-XML is missing `Baseline` elements. Use any script from `production/` or `updated/` — these all include baseline generation. The `experimental/` scripts do not.

**Dense vertical page detected as horizontal stripes.**
This is the case-study problem. Use the v5 script with `--strategy rotate90`. Raising resolution or tuning thresholds alone will not fix it; see the revised diagnosis above for why.

**rotate90 produces fragmented, overlapping boxes.**
The page has internally sparse or irregular columns (prefaces, honorific raising, blank gaps). Use `--strategy native` for such pages.

**rotate90 misses faint columns.**
Lower the score threshold: `--box_thresh 0.3`. For show-through pages, Sauvola preprocessing (`production/paddle_single_sauvola.py` logic) before detection also helps.

**Detection works on some pages but completely fails on others in the same manuscript.**
Pages with show-through or low contrast need preprocessing. Try `paddle_single_sauvola.py --preprocess binarize_sauvola`. If that introduces noise on high-contrast areas, try `paddle_single_selective_clahe.py --preprocess selective`.

**`use_textline_orientation=True` makes results worse.**
Expected on classical vertical layouts; the classifier was trained on modern text. v3 onward disables it by default; v5 exposes it as an opt-in flag for seg_rec mode only.

**Column ordering is wrong after import into eScriptorium.**
v5's column clustering handles most cases, including double registers. For 雙行夾註, the two sub-rows of a note share one column track and may interleave; correct these manually in eScriptorium, or segment notes as separate regions.

**`predict()` returns results but parsing fails / `dt_polys` is empty.**
Run `util_diagnose_paddle_api.py` and match script generation to version: 3.0.x → `production/`, 3.2.x → v2 to v4, any → v5.

**NumPy or OpenCV version conflict.**
Applies to the 3.0.x stack only: pin `numpy==1.26.4`. PaddleOCR 3.6.x works with current NumPy.

---

## References

- Wang, W., Xie, E., Li, X., Hou, W., Lu, T., Yu, G., Shao, S. (2019). Shape Robust Text Detection with Progressive Scale Expansion Network. CVPR 2019. https://arxiv.org/abs/1903.12473
- Liao, M., Wan, Z., Yao, C., Chen, K., Bai, X. (2020). Real-Time Scene Text Detection with Differentiable Binarization. AAAI 2020. https://arxiv.org/abs/1911.08947 (the detector family used by PaddleOCR)
- Yang, H., Jin, L., Huang, W., Yang, Z., Lai, S., Sun, J. (2018). Dense and Tight Detection of Chinese Characters in Historical Documents: Datasets and a Recognition Guided Detector. IEEE Access 6: 30174–30183. https://ieeexplore.ieee.org/document/8364534 ; datasets: https://github.com/HCIILAB/TKH_MTH_Datasets_Release
- AncientDoc: Benchmarking Vision-Language Models on Chinese Ancient Documents (identifies vertical typesetting, interlinear notes, and traditional characters as the defining OCR challenges of this material)
- CHURRO: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition. https://arxiv.org/abs/2509.19768
