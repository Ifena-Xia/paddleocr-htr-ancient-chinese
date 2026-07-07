# PaddleOCR HTR Pipeline for Ancient Chinese Manuscripts

Scripts that drive PaddleOCR to produce base segmentation of vertically-written ancient Chinese manuscripts, exporting PAGE-XML that imports into [eScriptorium](https://escriptorium.fr/).

Built as part of a handwritten text recognition (HTR) workflow for historical Chinese and Sino-Japanese manuscripts in the Penn Libraries digital collections.

---

## Which script do I use?

| Your situation | Use | Section |
|---|---|---|
| PaddleOCR 3.0.x installed | `production/paddle_single_with_baseline.py` or `production/paddle_batch_v1.py` | [production/](#production) |
| PaddleOCR 3.2.x or later, ordinary vertical pages | `updated/paddle_batch_v4_vertical_filter.py` | [updated/](#updated) |
| Dense columns, small-character commentary, 雙行夾註 | `updated/paddle_batch_v5_native_res.py --strategy rotate90` | [case study](#case-study-dense-vertical-layouts-in-xing-li-da-quan-shu) |
| Not sure what a manuscript needs | `utils/util_compare_det_strategies.py` on a few pages first | [utils/](#utils) |
| Uneven lighting, show-through, low contrast | `production/paddle_single_sauvola.py` | [production/](#production) |

After any script runs, import the `.xml` into eScriptorium: **Images → Import → Transcription (XML)**.

---

## Quick start

```bash
# Single image, segmentation only (3.0.x)
python3 production/paddle_single_with_baseline.py \
    --image your_page.jpg --outdir out --lang ch --to_pagexml

# Batch a folder (3.2.x)
python3 updated/paddle_batch_v4_vertical_filter.py \
    --input_dir images/ --outdir out --lang ch --to_pagexml

# Dense vertical layouts (read the case study first)
python3 updated/paddle_batch_v5_native_res.py \
    --image your_page.jpg --outdir out --strategy rotate90 --to_pagexml
```

---

## Background

eScriptorium's built-in segmentation performs poorly on vertically-written ancient Chinese. This pipeline runs PaddleOCR's detection engine as a first pass, producing bounding boxes and baselines in PAGE-XML that eScriptorium can import for manual correction and ground-truth production.

The scripts grew out of several months with two manuscripts: *Yōso zusetsu* (廱疽圖說), an illustrated Sino-Japanese medical treatise with irregular handwriting, and *Xing li da quan shu* (性理大全書), a densely-typeset Neo-Confucian compilation. The recurring problems were:

- PaddleOCR API changes across 3.0.0 to 3.2.x (parameter names, output structure, `ocr()` versus `predict()`).
- eScriptorium requires a `Baseline` element inside every `TextLine`. Omitting it causes silent import failure.
- Dense vertical layouts get read as horizontal text (see the case study).
- Show-through, illustrations, and low contrast need preprocessing before detection.

The `experimental/` folder keeps the debugging scripts. They are not for use; they record what was tried and why it was dropped.

## Test materials

Both manuscripts come from the Penn Libraries digital collections (Colenda Digital Repository) and are openly accessible:

- *Yōso zusetsu* (廱疽圖說), Sino-Japanese illustrated medical treatise, irregular handwriting: https://colenda.library.upenn.edu/catalog/81431-p3806r
- *Xing li da quan shu* (性理大全書), Neo-Confucian compilation, dense vertical typeset: https://colenda.library.upenn.edu/catalog/81431-p39k46864

---

## Repository structure

```
paddleocr-htr-ancient-chinese/
├── README.md
├── experimental/          # debugging history, not for production use
├── production/            # stable scripts for PaddleOCR 3.0.x
│   ├── paddle_single_with_baseline.py
│   ├── paddle_single_sauvola.py
│   ├── paddle_single_selective_clahe.py
│   └── paddle_batch_v1.py
├── updated/               # scripts for PaddleOCR 3.2.x and later
│   ├── paddle_batch_v2_predict_api.py
│   ├── paddle_batch_v3_no_orientation.py
│   ├── paddle_batch_v4_vertical_filter.py
│   └── paddle_batch_v5_native_res.py      # dense layouts, see case study
└── utils/
    ├── util_diagnose_paddle_api.py
    ├── util_merge_page_xml.py
    └── util_compare_det_strategies.py     # strategy benchmark, see case study
```

---

## Script reference

### production/

**`paddle_single_with_baseline.py`**
The first script whose PAGE-XML eScriptorium accepts. Each `TextLine` carries a `Baseline` set to the vertical midline between the top and bottom edges of the bounding box, which vertical text in eScriptorium requires and which earlier versions lacked. Uses the `ocr()` API with `det_db_*` parameters. PaddleOCR 3.0.x.

**`paddle_single_sauvola.py`**
Adds Sauvola binarization (`--preprocess binarize_sauvola`) for pages with uneven lighting, verso show-through, or low-contrast ink where detection misses lines. Sauvola (from `skimage`) suits historical documents better than Otsu or adaptive thresholding because it responds to local intensity variation. PaddleOCR 3.0.x.

**`paddle_single_selective_clahe.py`**
Adds CLAHE contrast enhancement in a selective mode that touches only low-contrast regions (found with a local standard-deviation filter) and leaves readable areas alone. Useful on mixed-contrast pages. Slow. PaddleOCR 3.0.x.

**`paddle_batch_v1.py`**
Folder version of `paddle_single_with_baseline.py`. Takes `--input_dir`; the OCR instance is built once and reused across images. PaddleOCR 3.0.x.

### updated/

v2 to v4 replace `ocr()` with `predict()`, parse the `OCRResult.json['res']` structure from PaddleOCR 3.2.x, and use the renamed init parameters (`text_det_*` for `det_db_*`).

**`paddle_batch_v2_predict_api.py`**
First working batch script for 3.2.x. Uses `use_textline_orientation=True`.

**`paddle_batch_v3_no_orientation.py`**
Same as v2 with `use_textline_orientation=False`. Orientation classification, trained on modern text, misread classical column layouts and worsened results, so this is the safer default.

**`paddle_batch_v4_vertical_filter.py`**
Adds `filter_vertical_boxes()`, which drops boxes whose height-to-width ratio falls below 1.8 (function default 1.5), and lowers detection thresholds for denser layouts. Fine on regular vertical pages; not enough for dense small-character pages (see case study). PaddleOCR 3.2.x+.

**`paddle_batch_v5_native_res.py`**
Adds native-resolution detection, a rotate-90 strategy, run modes (`seg` / `seg_rec` / `rec`), and column-aware reading order. Written for the dense-layout problem in *Xing li da quan shu*; documented in the case study. Validated under both the pinned 3.0.x environment and 3.6.x.

### utils/

**`util_diagnose_paddle_api.py`**
Run first when errors are unexpected. Prints the installed PaddleOCR version, confirms `PaddleOCR` initializes, and lists the parameters `ocr()` accepts, which tells you whether your version uses `ocr()` or `predict()`.

**`util_merge_page_xml.py`**
Merges a folder of PAGE-XML files into one, for consolidating results before import.

**`util_compare_det_strategies.py`**
Runs one page through three configurations (the v4-equivalent settings, native resolution, rotate-90) and writes overlay previews plus a stats table. This is the instrument behind the case study; run it on a few representative pages of any new manuscript before choosing a strategy.

### experimental/

Reference only. None of these should be used for production. The single most useful lesson from this phase: **eScriptorium silently discards `TextLine` elements with no `Baseline` child.** No error is raised on import; the segmentation seems to load, but the panel stays blank.

| Script | What it tried | Why superseded |
|--------|--------------|----------------|
| `exp_tesseract_baseline.py` | Tesseract instead of PaddleOCR | Poor on vertical ancient Chinese |
| `exp_paddle_v2v7_compat.py` | PaddleOCR 2.7 with partial 3.x shims | Unreliable on 3.x; superseded by v3_initial |
| `exp_paddle_v3_initial.py` | First 3.x attempt | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_tuned_params.py` | Tuned `det_db_*`, max_side 1200 | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_clahe_global.py` | Global CLAHE | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_clahe_selective.py` | Selective CLAHE | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_binarize_v1.py` | Adaptive/Otsu/Sauvola binarization; box filtering | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_binarize_v2.py` | Refined binarization; fallback on zero detections | No Baseline; eScriptorium rejected it |

---

## Case study: dense vertical layouts in *Xing li da quan shu*

v2 to v4 stay fully usable for the layouts they were built on. This section is about one manuscript property, densely packed vertical columns and small-character commentary, that defeats the standard configuration whatever the script version.

### The problem

On pages of *Xing li da quan shu* dominated by small characters (commentary in 雙行夾註 double-row interlinear format, or full pages of annotation), the detector returns wide horizontal boxes spanning many columns instead of one box per column. A human reads the columns without effort and the model recognizes the individual characters fine; only line formation fails.

### What v5 adds

`updated/paddle_batch_v5_native_res.py` provides:

- **Native-resolution detection.** Earlier scripts shrank every page to 1200 px before detection. v5 drops that and runs at native resolution (built-in cap 4000 px), with an emergency `--pre_max_side` flag for memory-limited machines.
- **Three strategies.** `native` (direct detection), `rotate90` (rotate the page 90 degrees so columns become horizontal lines, detect, then map coordinates back to the original image via a closed-form geometric inverse, `x = y_r, y = H−1−x_r`), and `auto` (run native, fall back to rotate90 when the vertical-box share is low, keep the better result).
- **Three run modes.** `seg` (detection only), `seg_rec` (detection plus recognition, text prefilled into the XML), and `rec` (read eScriptorium-corrected PAGE-XML, crop each line, recognize, write `TextEquiv` back, closing the loop).
- **Column-aware reading order.** Lines are clustered into columns by x-center, then sorted right-to-left, top-to-bottom within each column.
- **Calmer defaults for dense layouts.** `unclip_ratio=1.3` (v4 used 2.0, which dilates boxes into neighboring columns) and `box_thresh=0.45` (v4 used 0.2, which admits ruling lines and noise).

### Quick start for v5

```bash
# Dense small-character pages: rotate90 is the validated choice
python3 updated/paddle_batch_v5_native_res.py \
    --input_dir images/ --outdir out --strategy rotate90 --to_pagexml

# Irregular pages (prefaces, raised honorifics, mixed sizes): native
python3 updated/paddle_batch_v5_native_res.py \
    --image preface.jpg --outdir out --strategy native --to_pagexml

# Segmentation plus prefilled recognition
python3 updated/paddle_batch_v5_native_res.py \
    --input_dir images/ --outdir out --mode seg_rec --strategy rotate90 --to_pagexml

# Recognition only, backfilling eScriptorium-corrected XML
python3 updated/paddle_batch_v5_native_res.py \
    --input_dir images/ --xml_dir corrected_xml/ --outdir out --mode rec
```

Every flag is documented in the script header. Faint or show-through pages may need `--box_thresh 0.3` in rotate90 mode (see page 4 below).

### Validation results (five representative pages)

Five pages spanning the layout types of *Xing li da quan shu* were run through `util_compare_det_strategies.py`, comparing the v4-equivalent configuration (`legacy1200`), native-resolution detection (`native`), and `rotate90`. The V-ratio is the share of detected boxes with height-to-width ratio above 1.5.

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

Two findings stand out. First, on dense small-character pages (3, 4, 5), `rotate90` is the only strategy that yields vertical columns at all; `legacy1200` and `native` return horizontal stripes spanning the full text block, with median box widths of 500 to 760 px, essentially page-wide. Second, **native resolution alone barely moves the needle**: on pages 1, 3, and 4 the native results almost match legacy1200, and on page 5 they are marginally worse. The resolution-starvation hypothesis behind v5's design is therefore at most secondary. Rotation is the decisive intervention.

### Strategy selection guide

| Page type | Strategy |
|-----------|---------------------|
| Dense small-character pages, 雙行夾註 commentary, internally continuous columns | `rotate90` |
| Prefaces, irregular column starts, raised honorifics, large intra-column gaps | `native` |
| Faded or show-through dense pages | `rotate90 --box_thresh 0.3` |
| Unknown or mixed batch | `auto`, then inspect overlays |

A caution on `auto`: it switches on vertical-box counts, so it prefers whichever strategy produces more vertical boxes, which on irregular preface pages can favor rotate90's fragmented output over native's smaller but cleaner result. Page 1 of the validation set is exactly this case. Set the strategy explicitly for batches with known page types, and reserve `auto` for triage.

The deeper limit, shared by all strategies, is that pages mixing large main text with dense commentary (page 3) have no single best global setting. The principled fix is layout-first processing: detect text regions, classify them by density, and run the right strategy per region. That is the natural next step.

### Why this happens

PaddleOCR's detectors (the DB family) are segmentation-based. They predict a per-pixel text probability map, shrink each text instance to a kernel during training to keep neighbors apart, then re-expand detected kernels with an unclip ratio at inference. A known limitation of this family, stated in PSENet, an earlier detector in the same segmentation-based family, is that segmentation-based detectors "may not separate the text instances that are very close to each other" (Li et al. 2019). Yang et al. (2018), who built the TKH/MTH datasets for this exact problem, put the document-side version plainly: "Characters in historical documents are typically densely distributed" and resist standard detectors.

Proximity alone does not explain the results, and this is where the data forced a revision. If line formation were a neutral, isotropic function of pixel proximity, then raising resolution (native) should have separated the columns, and rotating the page should have changed nothing, since rotation preserves relative distances. The opposite happened. The horizontal stripes on page 4 show that the detector does not group text isotropically: it holds a strong learned prior for long horizontal line shapes, acquired from training data in which horizontal lines dominate. In a dense block where the horizontal gap between columns is comparable to or smaller than the vertical gap between characters within a column, that prior wins, and the model confidently reconstructs the layout as horizontal rows, one clean stripe per character row, exactly what the page 4 overlays show. Rotating the page 90 degrees does not change the geometry; it changes which axis the prior acts on. The true columns become horizontal lines, the modal case of the training distribution, and detection succeeds.

The account also explains where rotate90 fails. Columns that are internally sparse or irregular, such as page 1's preface with raised honorifics (抬頭), staggered column starts, and blank gaps, become short horizontal fragments at uneven offsets after rotation. The detector splits them at the gaps, and the fragments map back as overlapping boxes. Rotation helps when columns are internally continuous and densely stacked, and hurts when they are not. The missed columns in pages 3 and 4 under rotate90 are a separate, milder issue: faint ink and show-through depress the mean probability of a long line, and the default `box_thresh=0.45` prunes it. Lowering the threshold for faded pages recovers them.

To our knowledge no published study documents this rotation remedy, or the directional-prior account behind it, for DB-family detectors on vertical Chinese specifically. The explanation above is an inference from the detector's mechanics and from these validation results, consistent with the cited literature but not established in it. The closest published discussions are the dense-text detection literature (Yang et al. 2018; Li et al. 2019) and recent VLM benchmarks on Chinese ancient documents, which independently identify vertical typesetting, interlinear notes, and small characters as the defining challenges of this material. AncientDoc (Yu et al. 2025) states directly that Chinese ancient documents are vertically typeset right-to-left and that models must learn the correct reading order and in-column line breaks; the CHURRO report (Semnani et al. 2025) makes the broader point that VLMs built for modern standardized text fail on the irregular layouts and degradation of historical materials.

### Two manuscripts compared: why irregular handwriting segmented more easily than clean dense print

A counterintuitive observation: *Yōso zusetsu*, with its tilted columns, irregular handwriting, reproduction artifacts, and interleaved illustrations, was consistently easier to keep vertical in base segmentation than the cleanly carved, perfectly regular *Xing li da quan shu*.

The diagnosis above explains why. For this detector family, the difficulty of vertical base segmentation is set not by script neatness but by layout geometry: the relation between inter-column gaps and intra-column character spacing, and whether that relation lets the horizontal-line prior misfire. *Yōso zusetsu* has few columns per page, generously spaced, with brush flow that keeps each column continuous top to bottom; tilt and irregularity are no obstacle, because segmentation-based detectors handle arbitrary shapes well, which is what they were designed for. *Xing li da quan shu*'s small-character commentary inverts the geometry: columns jammed together, characters within a column separated by regular gaps, and the 雙行夾註 format actively pairing characters horizontally inside one column track. Visual noise is a recognition problem; gap geometry is a detection problem. The two manuscripts fail on different axes, and base segmentation lives on the second.

One practical corollary: a manuscript's suitability for this pipeline cannot be read off its apparent cleanliness. Run `util_compare_det_strategies.py` on a few pages first.

---

## Dependencies

The dependency story is layered by script generation. v5 was written against the PaddleOCR 3.6.0 source but uses only module APIs (`TextDetection`, `TextRecognition`, `PaddleOCR`) and parameter names stable across 3.0.x to 3.6.x, and it has been validated under both the pinned environment below and a current 3.6.x install.

| Scripts | PaddleOCR | Notes |
|---------|-----------|-------|
| `production/` | 3.0.x | `ocr()` API, `det_db_*` parameter names |
| `updated/` v2 to v4 | 3.2.x+ | `predict()` API, `text_det_*` parameter names |
| `updated/` v5, `utils/util_compare_det_strategies.py` | 3.0.x to 3.6.x | module APIs; validated under both |

**Pinned environment (tested, still works for everything here):**
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

**Current environment (starting fresh):**
```bash
python3 -m venv ~/venvs/paddleocr36-env
source ~/venvs/paddleocr36-env/bin/activate
pip install paddlepaddle paddleocr
```
The `numpy<2.0` pin is required for the 3.0.x stack; PaddleOCR 3.6.x resolved those incompatibilities.

Extra for `paddle_single_sauvola.py` and `selective_clahe.py`: `scikit-image`, `scipy`.

Model weights download on first run. If the default source is unreachable, set `PADDLE_PDX_MODEL_SOURCE=HUGGINGFACE`.

PaddleOCR updates often and can break APIs without deprecation warnings. On unexpected errors, run `util_diagnose_paddle_api.py` first.

---

## Known issues and troubleshooting

**eScriptorium import appears to succeed but the segmentation panel is blank.**
The PAGE-XML has no `Baseline` elements. Use any script from `production/` or `updated/`; they all generate baselines. The `experimental/` scripts do not.

**Dense vertical page detected as horizontal stripes.**
The case-study problem. Use v5 with `--strategy rotate90`. Raising resolution or tuning thresholds alone will not fix it; see the diagnosis above.

**rotate90 produces fragmented, overlapping boxes.**
The page has internally sparse or irregular columns (prefaces, honorific raising, blank gaps). Use `--strategy native`.

**rotate90 misses faint columns.**
Lower the score threshold: `--box_thresh 0.3`. For show-through pages, Sauvola preprocessing (`production/paddle_single_sauvola.py`) before detection also helps.

**Detection works on some pages, fails completely on others in the same manuscript.**
Show-through or low-contrast pages need preprocessing. Try `paddle_single_sauvola.py --preprocess binarize_sauvola`. If that adds noise on high-contrast areas, try `paddle_single_selective_clahe.py --preprocess selective`.

**`use_textline_orientation=True` makes results worse.**
Expected on classical vertical layouts; the classifier was trained on modern text. v3 onward disables it by default; v5 exposes it as an opt-in flag for seg_rec mode only.

**Column ordering is wrong after import into eScriptorium.**
v5's column clustering handles most cases, including double registers. For 雙行夾註, the two sub-rows of a note share one column track and may interleave; correct these by hand in eScriptorium, or segment notes as separate regions.

**`predict()` returns results but parsing fails or `dt_polys` is empty.**
Run `util_diagnose_paddle_api.py` and match script generation to version: 3.0.x → `production/`, 3.2.x → v2 to v4, any → v5.

**NumPy or OpenCV version conflict.**
Applies to the 3.0.x stack only: pin `numpy==1.26.4`. PaddleOCR 3.6.x works with current NumPy.

---

## References

- Li, X., Wang, W., Hou, W., Liu, R., Lu, T., Yang, J. (2019). Shape Robust Text Detection with Progressive Scale Expansion Network (PSENet). CVPR 2019. https://arxiv.org/abs/1806.02559
- Liao, M., Wan, Z., Yao, C., Chen, K., Bai, X. (2020). Real-Time Scene Text Detection with Differentiable Binarization (DB). AAAI 2020. https://arxiv.org/abs/1911.08947 (the detector family PaddleOCR uses)
- Yang, H., Jin, L., Huang, W., Yang, Z., Lai, S., Sun, J. (2018). Dense and Tight Detection of Chinese Characters in Historical Documents: Datasets and a Recognition Guided Detector. IEEE Access 6: 30174–30183. https://ieeexplore.ieee.org/document/8364534 ; datasets: https://github.com/HCIILAB/TKH_MTH_Datasets_Release
- Yu, H., et al. (2025). Benchmarking Vision-Language Models on Chinese Ancient Documents: From OCR to Knowledge Reasoning (AncientDoc). https://arxiv.org/abs/2509.09731
- Semnani, S., Zhang, H., He, X., Tekgürler, M., Lam, M. (2025). CHURRO: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition. EMNLP 2025. https://arxiv.org/abs/2509.19768
