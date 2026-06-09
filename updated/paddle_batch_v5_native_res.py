# paddle_batch_v5_native_res.py
# 適用版本：PaddleOCR 3.6.x（已對照 3.6.0 已安裝源碼核對全部參數名與輸出結構）
#
# 與 v4 的關鍵區別（為什麼性理大全的豎排在 v4 會失敗）：
#   v4 先用 PIL 把圖片硬縮到 max_side=1200 再送檢測。密排窄列在低解析度下
#   會被 DB 檢測模型黏連成橫排大框。PaddleOCR 3.6 的默認配置其實是
#   limit_type="min", limit_side_len=64, max_side_limit=4000，即以接近原始
#   解析度檢測。v5 腳本不再預縮放，並把 unclip_ratio 降到 1.3 以避免鄰列
#   膨脹黏連。另提供 rotate90 備用策略：把整頁旋轉 90 度後檢測（豎列變橫
#   行，是檢測模型最擅長的形態），再把座標精確映射回原圖。
#
# ============================== 用法總覽 ==============================
#
# 【模式控制 --mode】
#   seg      只做 base segmentation，只加載檢測模型（最輕量，推薦起步）
#   seg_rec  分割 + 文字識別，文字寫入 PAGE-XML 作預填
#   rec      只做識別：讀入 eScriptorium 修正後導出的 PAGE-XML，
#            按框裁切識別，把文字寫回新的 XML（需 --xml_dir）
#
# 【策略控制 --strategy】（僅 seg / seg_rec 模式）
#   native    直接以原始解析度檢測（默認，先試這個）
#   rotate90  整頁旋轉 90 度檢測後映射回原座標（密排頁的備用方案）
#   auto      先 native，若豎排框比例 < --auto_threshold 再試 rotate90，取較優
#
# 【XML 控制】
#   加 --to_pagexml 導出 PAGE-XML（eScriptorium 可導入，含 Baseline）
#   不加則只輸出 JSON 與疊框預覽圖
#
# 【示例】
#   單張，只分割，導出 XML：
#     python3 paddle_batch_v5_native_res.py --image page.jpg --outdir out --to_pagexml
#   批量，分割+識別，auto 策略：
#     python3 paddle_batch_v5_native_res.py --input_dir images/ --outdir out \
#         --mode seg_rec --strategy auto --to_pagexml
#   只識別（回填 eScriptorium 修正後的框）：
#     python3 paddle_batch_v5_native_res.py --input_dir images/ --xml_dir corrected_xml/ \
#         --outdir out --mode rec
#   記憶體緊張的舊 Mac（謹慎降解析度，2400 仍遠高於 v4 的 1200）：
#     python3 paddle_batch_v5_native_res.py --image page.jpg --outdir out \
#         --to_pagexml --pre_max_side 2400
#
# ======================================================================

import os, json, argparse, time, signal, glob
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# 全局實例，避免重複初始化
det_instance = None      # TextDetection（seg 模式）
ocr_instance = None      # PaddleOCR 完整管線（seg_rec 模式）
rec_instance = None      # TextRecognition（rec 模式）

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# ---------- 超時處理 ----------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超時")

def _alarm(seconds):
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
    except ValueError:
        pass  # 非主線程時跳過

def _alarm_off():
    try:
        signal.alarm(0)
    except ValueError:
        pass

# ---------- 圖片讀取（v5 不再默認縮放！） ----------
def load_image(image_path, pre_max_side=None):
    """
    v4 的 safe_resize(max_side=1200) 是豎排失敗的主因，已移除。
    pre_max_side 僅作為記憶體不足時的應急選項，默認 None = 原始解析度。
    PaddleOCR 3.6 管線內部已有 max_side_limit=4000 的保護。
    回傳: (PIL.Image RGB, scale)
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    scale = 1.0
    if pre_max_side is not None and max(img.size) > pre_max_side:
        w, h = img.size
        scale = pre_max_side / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"[WARN] 已按 --pre_max_side 縮放至 {img.size}（注意：縮放會損害密排豎列的分離）")
    return img, scale

# ---------- rotate90 座標映射 ----------
def rotate_img_cw(img):
    """整頁順時針旋轉 90 度。PIL 的 ROTATE_270 = 逆時針 270 = 順時針 90。"""
    return img.transpose(Image.ROTATE_270)

def map_point_back_from_cw(x_r, y_r, orig_w, orig_h):
    """
    順時針旋轉 90 度的正映射為 (x,y) -> (H-1-y, x)，H 為原圖高。
    逆映射（旋轉圖座標 -> 原圖座標）：x = y_r, y = H-1-x_r
    （已用合成圖逐點驗證，見 utils/util_compare_det_strategies.py）
    """
    return float(y_r), float(orig_h - 1 - x_r)

def map_polys_back_from_cw(polys, orig_w, orig_h):
    out = []
    for poly in polys:
        mapped = [map_point_back_from_cw(p[0], p[1], orig_w, orig_h) for p in poly]
        # 重排為左上起順時針四點，保持 Coords 與 Baseline 計算的一致性
        xs = [p[0] for p in mapped]; ys = [p[1] for p in mapped]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        out.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return out

# ---------- 豎排過濾（沿用 v4，可關閉） ----------
def filter_vertical_boxes(boxes, min_height_width_ratio=1.5):
    filtered = []
    for box in boxes:
        poly = box["poly"]
        if len(poly) < 4:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        width = max(xs) - min(xs); height = max(ys) - min(ys)
        if width == 0:
            filtered.append(box); continue
        if height / width > min_height_width_ratio:
            filtered.append(box)
        elif height > width * 1.2 and height > 20:
            filtered.append(box)
    print(f"[INFO] 豎排過濾：{len(boxes)} -> {len(filtered)}")
    return filtered

def vertical_ratio(boxes):
    """豎排框（高寬比>1.5）佔比，auto 策略據此判斷檢測是否成功"""
    if not boxes:
        return 0.0
    n = 0
    for b in boxes:
        xs = [p[0] for p in b["poly"]]; ys = [p[1] for p in b["poly"]]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        if w == 0 or h / w > 1.5:
            n += 1
    return n / len(boxes)

# ---------- 按列聚類的右到左排序（取代 v4 的簡單 x 降序） ----------
def sort_vertical_rtl(boxes):
    """
    先按 x 中心把框聚成列（間距閾值 = 中位框寬 * 0.8），
    列按右到左排，列內按 y 頂端從上到下排。
    對雙欄密排頁比單純 x 降序穩定得多。
    """
    if not boxes:
        return boxes
    def x_center(b):
        return sum(p[0] for p in b["poly"]) / len(b["poly"])
    def y_top(b):
        return min(p[1] for p in b["poly"])
    def width(b):
        xs = [p[0] for p in b["poly"]]
        return max(xs) - min(xs)

    widths = sorted(width(b) for b in boxes)
    med_w = widths[len(widths) // 2] if widths else 1
    gap = max(med_w * 0.8, 8)

    by_x = sorted(boxes, key=lambda b: -x_center(b))
    columns, current, last_x = [], [], None
    for b in by_x:
        xc = x_center(b)
        if last_x is None or abs(last_x - xc) <= gap:
            current.append(b)
        else:
            columns.append(current); current = [b]
        last_x = xc
    if current:
        columns.append(current)
    ordered = []
    for col in columns:
        ordered.extend(sorted(col, key=y_top))
    return ordered

# ---------- 檢測（seg 模式：只加載檢測模型） ----------
def init_det(args):
    global det_instance
    if det_instance is not None:
        return det_instance
    from paddleocr import TextDetection
    print(f"[INFO] 初始化 TextDetection（{args.det_model}）...")
    _alarm(300)
    det_instance = TextDetection(
        model_name=args.det_model,
        limit_side_len=args.det_limit_side_len,
        limit_type="min",
        thresh=args.det_thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip,
        device=args.device,
    )
    _alarm_off()
    print("[INFO] TextDetection 初始化成功")
    return det_instance

def init_ocr(args):
    global ocr_instance
    if ocr_instance is not None:
        return ocr_instance
    from paddleocr import PaddleOCR
    print("[INFO] 初始化 PaddleOCR 完整管線（檢測+識別）...")
    _alarm(300)
    ocr_instance = PaddleOCR(
        lang=args.lang,
        text_detection_model_name=args.det_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=args.use_textline_orientation,
        text_det_limit_side_len=args.det_limit_side_len,
        text_det_limit_type="min",
        text_det_thresh=args.det_thresh,
        text_det_box_thresh=args.box_thresh,
        text_det_unclip_ratio=args.unclip,
        device=args.device,
    )
    _alarm_off()
    print("[INFO] PaddleOCR 初始化成功")
    return ocr_instance

def init_rec(args):
    global rec_instance
    if rec_instance is not None:
        return rec_instance
    from paddleocr import TextRecognition
    print("[INFO] 初始化 TextRecognition...")
    _alarm(300)
    rec_instance = TextRecognition(device=args.device)
    _alarm_off()
    return rec_instance

def run_detection(img, args):
    """對單張 PIL 圖跑一次檢測或檢測+識別，回傳 boxes 列表"""
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    boxes = []
    _alarm(args.timeout)
    try:
        if args.mode == "seg":
            det = init_det(args)
            results = det.predict(input=arr, batch_size=1)
            for res in results:
                data = res.json.get("res", res.json)
                polys = data.get("dt_polys", [])
                scores = data.get("dt_scores", [1.0] * len(polys))
                for poly, sc in zip(polys, scores):
                    poly = np.asarray(poly, dtype=float).tolist()
                    boxes.append({"poly": poly, "text": "", "score": float(sc)})
        else:  # seg_rec
            ocr = init_ocr(args)
            results = ocr.predict(arr)
            for res in results:
                data = res.json.get("res", res.json)
                polys = data.get("rec_polys") or data.get("dt_polys") or []
                texts = data.get("rec_texts", [""] * len(polys))
                scores = data.get("rec_scores", [1.0] * len(polys))
                for poly, txt, sc in zip(polys, texts, scores):
                    poly = np.asarray(poly, dtype=float).tolist()
                    boxes.append({"poly": poly, "text": txt, "score": float(sc)})
    except TimeoutException:
        print(f"[ERROR] 檢測超時（{args.timeout}s）")
    finally:
        _alarm_off()
    return boxes

def detect_with_strategy(img, args):
    """native / rotate90 / auto 三種策略"""
    W, H = img.size
    native_boxes = []
    if args.strategy in ("native", "auto"):
        native_boxes = run_detection(img, args)
        vr = vertical_ratio(native_boxes)
        print(f"[INFO] native：{len(native_boxes)} 框，豎排佔比 {vr:.0%}")
        if args.strategy == "native" or (native_boxes and vr >= args.auto_threshold):
            return native_boxes, "native"
        print(f"[INFO] 豎排佔比低於 {args.auto_threshold:.0%}，auto 切換 rotate90")

    img_r = rotate_img_cw(img)
    boxes_r = run_detection(img_r, args)
    polys = map_polys_back_from_cw([b["poly"] for b in boxes_r], W, H)
    boxes90 = [{"poly": p, "text": b["text"], "score": b["score"]}
               for p, b in zip(polys, boxes_r)]
    vr90 = vertical_ratio(boxes90)
    print(f"[INFO] rotate90：{len(boxes90)} 框，映射回原圖後豎排佔比 {vr90:.0%}")
    if args.strategy == "rotate90":
        return boxes90, "rotate90"
    # auto：比較兩種策略，取豎排框絕對數量較多者
    def n_vert(bs):
        return int(round(vertical_ratio(bs) * len(bs)))
    chosen = (boxes90, "rotate90") if n_vert(boxes90) >= n_vert(native_boxes) else (native_boxes, "native")
    print(f"[INFO] auto 最終選擇：{chosen[1]}")
    return chosen

# ---------- rec 模式：讀 PAGE-XML，裁切識別，回填 ----------
def crop_line(img_arr, poly):
    """min-area 裁切；高瘦豎列旋轉 90 度後送識別（PaddleOCR 識別模型的訓練分佈）"""
    import cv2
    pts = np.asarray(poly, dtype=np.float32)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), img_arr.shape[1]), min(int(y1), img_arr.shape[0])
    crop = img_arr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h >= w * 1.5:
        crop = np.rot90(crop, k=-1)  # 順時針 90
    return crop

def rec_from_pagexml(image_path, xml_path, args):
    rec = init_rec(args)
    img, _ = load_image(image_path, args.pre_max_side)
    arr = np.array(img)[:, :, ::-1]
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"p": PAGE_NS}
    n_done = 0
    for line in root.iter(f"{{{PAGE_NS}}}TextLine"):
        coords = line.find(f"{{{PAGE_NS}}}Coords")
        if coords is None:
            continue
        poly = [[float(v) for v in pt.split(",")] for pt in coords.get("points", "").split()]
        crop = crop_line(arr, poly)
        if crop is None:
            continue
        try:
            out = rec.predict(input=crop, batch_size=1)
            for r in out:
                data = r.json.get("res", r.json)
                text = data.get("rec_text", "")
                score = data.get("rec_score", 0.0)
                # 移除舊 TextEquiv 再寫入
                for old in line.findall(f"{{{PAGE_NS}}}TextEquiv"):
                    line.remove(old)
                te = ET.SubElement(line, f"{{{PAGE_NS}}}TextEquiv", {"conf": f"{score:.4f}"})
                ET.SubElement(te, f"{{{PAGE_NS}}}Unicode").text = text
                n_done += 1
        except Exception as e:
            print(f"[WARN] 單行識別失敗：{e}")
    print(f"[INFO] 已回填 {n_done} 行文字")
    return tree

# ---------- PAGE-XML 導出（結構與 v4 完全一致，eScriptorium 已驗證可導入） ----------
def to_pagexml(image_path, img_size, boxes_sorted, save_xml, with_rec=False):
    W, H = img_size
    root = ET.Element("PcGts", {
        "xmlns": PAGE_NS,
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"
    })
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "PaddleOCR HTR Pipeline v5"
    ET.SubElement(metadata, "Created").text = time.strftime("%Y-%m-%dT%H:%M:%S")
    page = ET.SubElement(root, "Page", {
        "imageFilename": os.path.basename(image_path),
        "imageWidth": str(int(W)), "imageHeight": str(int(H))
    })
    region = ET.SubElement(page, "TextRegion", {"id": "r1"})
    ET.SubElement(region, "Coords", {"points": f"0,0 {int(W)},0 {int(W)},{int(H)} 0,{int(H)}"})
    for i, box in enumerate(boxes_sorted, 1):
        if not box["poly"] or len(box["poly"]) != 4:
            continue
        line = ET.SubElement(region, "TextLine", {"id": f"l{i}"})
        pts = " ".join(f"{int(x)},{int(y)}" for (x, y) in box["poly"])
        ET.SubElement(line, "Coords", {"points": pts})
        # Baseline：豎排基線，上邊中點 -> 下邊中點（eScriptorium 必需，否則靜默丟棄）
        poly = box["poly"]
        x_top = int((poly[0][0] + poly[1][0]) // 2)
        y_top = int((poly[0][1] + poly[1][1]) // 2)
        x_bot = int((poly[2][0] + poly[3][0]) // 2)
        y_bot = int((poly[2][1] + poly[3][1]) // 2)
        ET.SubElement(line, "Baseline", {"points": f"{x_top},{y_top} {x_bot},{y_bot}"})
        if with_rec and box.get("text"):
            te = ET.SubElement(line, "TextEquiv")
            ET.SubElement(te, "Unicode").text = box["text"]
    tree = ET.ElementTree(root)
    tree.write(save_xml, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] PAGE-XML 已保存：{save_xml}")

# ---------- 預覽圖 ----------
def save_overlay(img, boxes, path):
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    for b in boxes:
        pts = [(p[0], p[1]) for p in b["poly"]]
        d.polygon(pts, outline=(220, 40, 40), width=4)
    vis.save(path)
    print(f"[INFO] 預覽圖已保存：{path}")

# ---------- 主流程 ----------
def process_image(image_path, args):
    stem = Path(image_path).stem
    print(f"\n[INFO] 處理 {image_path}")
    if args.mode == "rec":
        xml_in = os.path.join(args.xml_dir, stem + ".xml")
        if not os.path.exists(xml_in):
            print(f"[WARN] 找不到對應 XML：{xml_in}，跳過")
            return
        tree = rec_from_pagexml(image_path, xml_in, args)
        out_xml = os.path.join(args.outdir, stem + "_rec.xml")
        tree.write(out_xml, encoding="utf-8", xml_declaration=True)
        print(f"[OK] 識別結果已寫入：{out_xml}")
        return

    img, scale = load_image(image_path, args.pre_max_side)
    boxes, used = detect_with_strategy(img, args)

    if not args.no_vertical_filter:
        boxes = filter_vertical_boxes(boxes, args.min_hw_ratio)
    boxes = sort_vertical_rtl(boxes)

    # 預覽圖在工作解析度上繪製（此時座標與圖一致）
    save_overlay(img, boxes, os.path.join(args.outdir, stem + "_overlay.png"))

    # 若使用了應急縮放，把座標還原到原圖尺寸再寫 JSON / XML
    if scale != 1.0:
        inv = 1.0 / scale
        for b in boxes:
            b["poly"] = [[p[0] * inv, p[1] * inv] for p in b["poly"]]
        img_size = Image.open(image_path).size
    else:
        img_size = img.size

    out_json = os.path.join(args.outdir, stem + ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"strategy": used, "n_boxes": len(boxes), "boxes": boxes},
                  f, ensure_ascii=False, indent=1)
    if args.to_pagexml:
        to_pagexml(image_path, img_size, boxes,
                   os.path.join(args.outdir, stem + ".xml"),
                   with_rec=(args.mode == "seg_rec"))

def main():
    ap = argparse.ArgumentParser(description="PaddleOCR 3.6.x 豎排古籍 base segmentation（原生解析度版）")
    ap.add_argument("--image", help="單張圖片路徑")
    ap.add_argument("--input_dir", help="批量處理的圖片文件夾")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["seg", "seg_rec", "rec"], default="seg")
    ap.add_argument("--strategy", choices=["native", "rotate90", "auto"], default="auto")
    ap.add_argument("--auto_threshold", type=float, default=0.6,
                    help="auto 策略下 native 結果豎排佔比低於此值則嘗試 rotate90")
    ap.add_argument("--to_pagexml", action="store_true")
    ap.add_argument("--xml_dir", help="rec 模式：eScriptorium 導出 XML 所在文件夾")
    ap.add_argument("--lang", default="ch", help="ch 同時覆蓋簡繁；PP-OCRv5 原生支持繁體")
    ap.add_argument("--det_model", default="PP-OCRv5_server_det",
                    help="記憶體緊張可改 PP-OCRv5_mobile_det")
    ap.add_argument("--det_limit_side_len", type=int, default=64,
                    help="3.6 默認 64 配合 limit_type=min，即不縮小圖片")
    ap.add_argument("--det_thresh", type=float, default=0.3)
    ap.add_argument("--box_thresh", type=float, default=0.45,
                    help="v4 的 0.2 噪聲過多；密排頁建議 0.4-0.5")
    ap.add_argument("--unclip", type=float, default=1.3,
                    help="v4 的 2.0 會把鄰列黏連；密排頁建議 1.2-1.5")
    ap.add_argument("--use_textline_orientation", action="store_true",
                    help="seg_rec 模式下若識別出現大量顛倒文字再開啟")
    ap.add_argument("--pre_max_side", type=int, default=None,
                    help="僅在記憶體不足時使用；會降低密排列分離能力")
    ap.add_argument("--no_vertical_filter", action="store_true")
    ap.add_argument("--min_hw_ratio", type=float, default=1.5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--device", default="cpu", help="cpu / gpu:0")
    args = ap.parse_args()

    if args.mode == "rec" and not args.xml_dir:
        ap.error("rec 模式需要 --xml_dir")
    if not args.image and not args.input_dir:
        ap.error("需要 --image 或 --input_dir")

    os.makedirs(args.outdir, exist_ok=True)
    if args.image:
        process_image(args.image, args)
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        files = sorted(sum([glob.glob(os.path.join(args.input_dir, e)) for e in exts], []))
        print(f"[INFO] 共 {len(files)} 張圖片")
        for fp in files:
            try:
                process_image(fp, args)
            except Exception as e:
                print(f"[ERROR] {fp} 處理失敗：{e}")

if __name__ == "__main__":
    main()
