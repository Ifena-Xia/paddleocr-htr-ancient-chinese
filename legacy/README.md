# Legacy

Earlier working versions, superseded by `current/` and kept for lineage. All emit baselines and imported cleanly in their day.

- `paddle_batch_v1.py` is the 2.x-era original. It relies on 2.x calling conventions (`cls=True`, `use_dilation`, the classic `.ocr()` output), which 3.x constructor validation rejects, so it does not run on a 3.x install.
- `paddle_batch_v2_predict_api.py` and `paddle_batch_v3_no_orientation.py` run on 3.2.x and import cleanly, but `current/`'s v4 supersedes them. v2 was the first working 3.2.x batch script; v3 turned off the orientation classifier; v4 keeps v3's settings and adds the vertical filter. Lineage v2 → v3 → v4.
- `paddle_single_*` are the single-image preprocessing predecessors (Sauvola, selective CLAHE, baseline export) later folded into `current/`'s v5.

For the full breakdown, see the main README.
