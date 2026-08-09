# Experimental

- The original scripts, written against PaddleOCR 2.x. They stopped working once PaddleOCR 3.x tightened its constructor validation: the 2.x calling conventions (`cls=True`, `use_dilation`, and the classic `.ocr()` output format) are rejected or misparsed on any 3.x install.

- `paddle_batch_v2_predict_api.py` and `paddle_batch_v3_no_orientation.py` work and import cleanly, but v4 supersedes them. v2 was the first working 3.2.x batch script; v3 turned off the orientation classifier; v4 keeps v3's settings and adds the vertical filter. They are kept so the lineage v2 to v3 to v4 stays legible.

For the full breakdown, see the main README.
