# Experimental

Development trail. Nothing here is for use. These scripts share one defect: they never produced importable output. eScriptorium silently discards any `TextLine` that has no `Baseline` child, so the import appears to succeed while the segmentation panel stays blank, and every `exp_*` script here lacks that element. Two fail for a further reason: `exp_tesseract_baseline.py` uses Tesseract, which is poor on vertical ancient Chinese, and `exp_paddle_v2v7_compat.py` mixes 2.7 conventions with 3.x shims and is unreliable on any 3.x install.

Kept for provenance only. Use `current/` instead.
