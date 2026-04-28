# PaddleOCR HTR Pipeline for Ancient Chinese Manuscripts

Scripts for driving PaddleOCR to generate base segmentation of vertically-written ancient Chinese manuscripts, producing PAGE-XML output compatible with [eScriptorium](https://escriptorium.fr/).

Developed as part of a handwritten text recognition (HTR) workflow for historical Chinese and Sino-Japanese manuscripts held in the Penn Libraries digital collections.

---

## Background and motivation

eScriptorium's built-in segmentation performs poorly on vertically-written ancient Chinese. This pipeline uses PaddleOCR's detection engine as a first pass to generate bounding boxes and baselines in PAGE-XML format, which can then be imported into eScriptorium for manual correction and ground truth production.

The scripts evolved over several months of working with two manuscripts: *Yōso zusetsu* (廱疽圖說), an illustrated Sino-Japanese medical treatise with irregular handwriting, and *Xing li da quan shu* (性理大全書), a densely-typeset Neo-Confucian compilation. The key challenges encountered were:

- PaddleOCR API breaking changes across 3.0.0 → 3.2.x (parameter names, output structure, `ocr()` vs `predict()`)
- eScriptorium requiring a `Baseline` element inside each `TextLine` — missing this causes silent import failure
- Dense vertical layouts where text columns are spaced so closely that the model interprets them as horizontal text
- Pages with show-through, illustrations, or very low contrast requiring preprocessing before detection

The `experimental/` folder preserves the scripts written during debugging. They are not recommended for use but document what was tried and why it was abandoned.

## Test materials

Scripts were developed and tested on two manuscripts from the
Penn Libraries digital collections (Colenda Digital Repository),
both openly accessible:

- *Yōso zusetsu* (廱疽圖說): Sino-Japanese illustrated medical
  treatise, irregular handwriting
  https://colenda.library.upenn.edu/catalog/81431-p3806r

- *Xing li da quan shu* (性理大全書): Neo-Confucian compilation,
  dense vertical typeset layout
  https://colenda.library.upenn.edu/catalog/81431-p39k46864

---

## Quick start

If you are using **PaddleOCR 3.2.x or later**, start with `updated/paddle_batch_v4_vertical_filter.py`.

If you are using **PaddleOCR 3.0.0**, start with `production/paddle_single_with_baseline.py` or `production/paddle_batch_v1.py`.

```bash
# Single image, segmentation only
python3 production/paddle_single_with_baseline.py \
    --image your_page.jpg --outdir out --lang ch --to_pagexml

# Single image, segmentation + recognition text prefilled
python3 production/paddle_single_with_baseline.py \
    --image your_page.jpg --outdir out --lang ch --to_pagexml --with_rec

# Batch processing a folder (PaddleOCR 3.2.x)
python3 updated/paddle_batch_v4_vertical_filter.py \
    --input_dir images/ --outdir out --lang ch --to_pagexml
```

Then import the `.xml` output into eScriptorium: **Images → Import → Transcription (XML)**.

---

## Repository structure

```
paddleocr-htr-scripts/
├── README.md
├── experimental/          # Debugging history — not for production use
│   ├── exp_tesseract_baseline.py
│   ├── exp_paddle_v2v7_compat.py
│   ├── exp_paddle_v3_initial.py
│   ├── exp_paddle_v3_tuned_params.py
│   ├── exp_paddle_v3_clahe_global.py
│   ├── exp_paddle_v3_clahe_selective.py
│   ├── exp_paddle_v3_binarize_v1.py
│   └── exp_paddle_v3_binarize_v2.py
├── production/            # Stable scripts for PaddleOCR 3.0.x
│   ├── paddle_single_with_baseline.py
│   ├── paddle_single_sauvola.py
│   ├── paddle_single_selective_clahe.py
│   └── paddle_batch_v1.py
├── updated/               # Updated scripts for PaddleOCR 3.2.x+
│   ├── paddle_batch_v2_predict_api.py
│   ├── paddle_batch_v3_no_orientation.py
│   └── paddle_batch_v4_vertical_filter.py
└── utils/
    ├── util_diagnose_paddle_api.py
    └── util_merge_page_xml.py
```

---

## Script reference

### production/

**`paddle_single_with_baseline.py`**
The first script that produces PAGE-XML accepted by eScriptorium. Key feature: each `TextLine` element includes a `Baseline` computed as the midpoint line between the top and bottom edges of the bounding box — this is required for vertical text in eScriptorium and was absent in all earlier versions. Uses `ocr()` API with `det_db_*` parameters. PaddleOCR 3.0.x.

**`paddle_single_sauvola.py`**
Adds Sauvola binarization preprocessing (`--preprocess binarize_sauvola`). Use this for pages with uneven lighting, show-through from the verso, or low-contrast ink where standard detection misses lines. The Sauvola algorithm (from `skimage`) is better suited to historical documents than Otsu or adaptive thresholding because it accounts for local intensity variation. PaddleOCR 3.0.x.

**`paddle_single_selective_clahe.py`**
Adds CLAHE contrast enhancement with a selective mode that applies enhancement only to low-contrast regions of the image (using a local standard deviation filter), leaving high-contrast regions untouched. Useful when a page has mixed contrast — enhancing the whole image uniformly can introduce noise into already-readable areas. Heavy time cost. PaddleOCR 3.0.x.

**`paddle_batch_v1.py`**
Batch-processing version of `paddle_single_with_baseline.py`. Takes `--input_dir` to process a whole folder at once. The OCR instance is initialized once and reused across all images. PaddleOCR 3.0.x.

### updated/

These three scripts replace `ocr()` with `predict()` and parse the new `OCRResult.json['res']` output structure introduced in PaddleOCR 3.2.x. They also use the renamed initialization parameters (`text_det_*` instead of `det_db_*`).

**`paddle_batch_v2_predict_api.py`**
First working batch script for PaddleOCR 3.2.x. Uses `use_textline_orientation=True`.

**`paddle_batch_v3_no_orientation.py`**
Same as v2 but with `use_textline_orientation=False`. In practice, enabling orientation classification on ancient vertical Chinese worsened results — the classifier was trained on modern text and misread classical column layouts. This is the safer default.

**`paddle_batch_v4_vertical_filter.py`** ← **recommended for 3.2.x+**
Adds a `filter_vertical_boxes()` post-processing step that discards detected boxes with height-to-width ratio below 1.5, removing false positives from illustrations and horizontal noise. Also lowers detection thresholds (`text_det_thresh=0.2`, `text_det_unclip_ratio=2.0`) for denser layouts. Recommended starting point for new work on PaddleOCR 3.2.x.

### utils/

**`util_diagnose_paddle_api.py`**
Run this first if you are getting unexpected errors. It prints the installed PaddleOCR version, confirms the `PaddleOCR` class initializes correctly, and lists the parameters accepted by `ocr()`. Useful for confirming whether you are on a version that uses `ocr()` or `predict()`.

**`util_merge_page_xml.py`**
Merges multiple PAGE-XML files from a folder into a single XML file. Useful for consolidating segmentation results before import.

### experimental/

These scripts are preserved for reference only. None of them should be used for production work. Brief notes on what each attempted and why it was superseded:

| Script | What it tried | Why superseded |
|--------|--------------|----------------|
| `exp_tesseract_baseline.py` | Tesseract as an alternative to PaddleOCR | Poor performance on vertical ancient Chinese |
| `exp_paddle_v2v7_compat.py` | PaddleOCR 2.7 with partial 3.x compatibility shims | Unreliable on 3.x; superseded by v3_initial |
| `exp_paddle_v3_initial.py` | First attempt at PaddleOCR 3.x, using `predict()` + `ocr()` fallback | No Baseline in XML output; eScriptorium rejected it |
| `exp_paddle_v3_tuned_params.py` | Tuned `det_db_*` detection parameters, max_side 1200 | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_clahe_global.py` | Added global CLAHE contrast enhancement | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_clahe_selective.py` | Added selective CLAHE (low-contrast regions only) | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_binarize_v1.py` | Added adaptive, Otsu, and Sauvola binarization options; `filter_boxes()` post-processing | No Baseline; eScriptorium rejected it |
| `exp_paddle_v3_binarize_v2.py` | Refined binarization parameters; added fallback to no-preprocessing if detection returns zero boxes | No Baseline; eScriptorium rejected it |

The single most important lesson from this entire experimental phase: **eScriptorium silently discards `TextLine` elements that do not contain a `Baseline` child element.** No error is raised on import. The segmentation appears to load, but the panel remains blank. This took considerable time to diagnose.

---

## Dependencies

**For production/ and updated/ scripts:**
```
paddlepaddle>=3.0.0
paddleocr>=3.0.0
numpy<2.0        # PaddleOCR has had silent incompatibilities with NumPy 2.x
opencv-python
Pillow
```

**Additional for paddle_single_sauvola.py and selective_clahe.py:**
```
scikit-image
scipy           # selective_clahe only
```

**Recommended virtual environment setup (tested combination):**
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

PaddleOCR updates frequently and often introduces breaking changes without deprecation warnings. If you encounter unexpected errors, run `util_diagnose_paddle_api.py` first to confirm your environment, and check whether parameter names have changed in the current release.

---

## Known issues and troubleshooting

**eScriptorium import appears to succeed but the segmentation panel is blank.**
The PAGE-XML is missing `Baseline` elements. Use any script from `production/` or `updated/` — these all include baseline generation. The `experimental/` scripts do not.

**`predict()` returns results but `dt_polys` is empty.**
The output structure changed between PaddleOCR minor versions. Run `util_diagnose_paddle_api.py` and check which version you have. If you are on 3.0.x, use `ocr()` (production scripts). If on 3.2.x+, use `predict()` (updated scripts).

**Detection misses lines on pages with dense vertical layout.**
Lower `text_det_thresh` and raise `text_det_unclip_ratio`. `paddle_batch_v4_vertical_filter.py` uses `thresh=0.2` and `unclip_ratio=2.0` which worked for *Xing li da quan shu*. Be aware that lowering the threshold too far will pick up illustration borders and ruling lines.

**Detection works on some pages but completely fails on others in the same manuscript.**
Pages with show-through or low contrast need preprocessing. Try `paddle_single_sauvola.py --preprocess binarize_sauvola`. If that introduces noise on high-contrast areas, try `paddle_single_selective_clahe.py --preprocess selective`.

**`use_textline_orientation=True` makes results worse.**
Expected. The orientation classifier was trained on modern Chinese documents and misidentifies classical vertical column layouts. Use `paddle_batch_v3_no_orientation.py` or `paddle_batch_v4_vertical_filter.py`, which disable it.

**Column ordering is wrong after import into eScriptorium.**
The `sort_vertical_rtl()` function sorts by x-center descending (right to left) then y-top ascending (top to bottom within each column). For double-column pages where columns are very close together, automatic ordering often fails. The reliable fix is to manually unlink lines from their region in eScriptorium and relink them to separate regions, one per column.

**`NumPy` or `OpenCV` version conflict.**
PaddleOCR silently breaks with NumPy 2.x. Pin to `numpy==1.26.4` in your virtual environment. See the recommended setup above.
