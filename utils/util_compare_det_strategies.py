# util_compare_det_strategies.py
# 診斷工具：在同一張掃描頁上對比三種檢測策略，輸出疊框預覽圖與統計表，
# 用於在真實掃描件（性理大全等）上驗證哪種策略最適合該古籍。
#
# 對比的三種策略：
#   legacy1200  模擬 v4 的做法（先縮到 1200px，thresh=0.2, unclip=2.0）
#   native      v5 默認（原生解析度，unclip=1.3, box_thresh=0.45）
#   rotate90    整頁旋轉 90 度檢測後映射回原座標
#
# 用法：
#   python3 util_compare_det_strategies.py --image page.jpg --outdir compare_out
#
# 輸出：
#   compare_out/<stem>_legacy1200.png / _native.png / _rotate90.png  疊框預覽
#   compare_out/<stem>_stats.json                                    統計表
# 統計指標：框總數、豎排框（高寬比>1.5）數量與佔比、框寬中位數。
# 期望：對密排豎排頁，native 與 rotate90 的豎排框數應顯著高於 legacy1200。

import os, json, argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

def stats(polys):
    n_vert, widths = 0, []
    for poly in polys:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        widths.append(w)
        if w == 0 or h / w > 1.5:
            n_vert += 1
    med_w = sorted(widths)[len(widths) // 2] if widths else 0
    return {"n_boxes": len(polys), "n_vertical": n_vert,
            "vertical_ratio": round(n_vert / len(polys), 3) if polys else 0.0,
            "median_box_width": round(float(med_w), 1)}

def overlay(img, polys, path, color=(220, 40, 40)):
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    for poly in polys:
        d.polygon([(p[0], p[1]) for p in poly], outline=color, width=4)
    vis.save(path)

def detect(arr, **det_kwargs):
    from paddleocr import TextDetection
    det = TextDetection(model_name=det_kwargs.pop("model_name", "PP-OCRv5_server_det"),
                        **det_kwargs)
    polys = []
    for res in det.predict(input=arr, batch_size=1):
        data = res.json.get("res", res.json)
        for poly in data.get("dt_polys", []):
            polys.append(np.asarray(poly, dtype=float).tolist())
    return polys

def map_back_cw(polys, orig_w, orig_h):
    out = []
    for poly in polys:
        mapped = [[float(p[1]), float(orig_h - 1 - p[0])] for p in poly]
        xs = [p[0] for p in mapped]; ys = [p[1] for p in mapped]
        out.append([[min(xs), min(ys)], [max(xs), min(ys)],
                    [max(xs), max(ys)], [min(xs), max(ys)]])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    stem = Path(args.image).stem

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    results = {}

    # ---- legacy1200：重現 v4 行為 ----
    small = img.copy()
    if max(W, H) > 1200:
        s = 1200 / max(W, H)
        small = img.resize((int(W * s), int(H * s)), Image.LANCZOS)
    else:
        s = 1.0
    arr = np.array(small)[:, :, ::-1]
    polys = detect(arr, thresh=0.2, box_thresh=0.2, unclip_ratio=2.0,
                   limit_side_len=1200, limit_type="max")
    polys = [[[p[0] / s, p[1] / s] for p in poly] for poly in polys]
    results["legacy1200"] = stats(polys)
    overlay(img, polys, os.path.join(args.outdir, f"{stem}_legacy1200.png"))

    # ---- native：v5 默認 ----
    arr = np.array(img)[:, :, ::-1]
    polys = detect(arr, thresh=0.3, box_thresh=0.45, unclip_ratio=1.3,
                   limit_side_len=64, limit_type="min")
    results["native"] = stats(polys)
    overlay(img, polys, os.path.join(args.outdir, f"{stem}_native.png"))

    # ---- rotate90 ----
    arr_r = np.array(img.transpose(Image.ROTATE_270))[:, :, ::-1]
    polys_r = detect(arr_r, thresh=0.3, box_thresh=0.45, unclip_ratio=1.3,
                     limit_side_len=64, limit_type="min")
    polys = map_back_cw(polys_r, W, H)
    results["rotate90"] = stats(polys)
    overlay(img, polys, os.path.join(args.outdir, f"{stem}_rotate90.png"))

    out = os.path.join(args.outdir, f"{stem}_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\n[INFO] 統計與預覽圖已保存至 {args.outdir}")
    print("[INFO] 判讀：豎排框數（n_vertical）多且預覽圖中每列一框者為優")

if __name__ == "__main__":
    main()
