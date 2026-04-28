# 用法：
#  只做分割：python3 exp_paddle_v3_initial.py --image Page_3.png --outdir out --lang ch --to_pagexml
#  分割+文字：python3 exp_paddle_v3_initial.py --image Page_3.png --outdir out --lang ch --to_pagexml --with_rec

import os, json, argparse, time, signal
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from paddleocr import PaddleOCR
import xml.etree.ElementTree as ET

# 全局OCR實例，避免重複初始化
ocr_instance = None

# ---------- 超時處理 ----------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

# ---------- 防崩潰：過大圖片先縮邊 ----------
def safe_resize(image_path, max_side=800):  # 減小最大尺寸
    img = Image.open(image_path)
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] 圖片縮小為 {new_w}x{new_h} (縮放比例: {scale:.3f})")
    return img, scale

# ---------- 偵測（可丟文字） ----------
def detect_boxes(image_path, lang='ch', drop_rec=True):
    """
    drop_rec=True  → 只要框（做 base segmentation 初稿）
    drop_rec=False → 框 + 文字（把文字也寫進 PAGE-XML 當預填）
    回傳: list[dict]，每個元素: {"poly": [[x1,y1]..[x4,y4]], "text": str, "score": float}
    """
    global ocr_instance
    
    # 初始化OCR實例（仅一次）
    if ocr_instance is None:
        print("[INFO] 初始化PaddleOCR...")
        try:
            # 設置超時處理
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 2分鐘超時
            
            # 使用最新的API參數
            ocr_instance = PaddleOCR(
                lang=lang,
                use_textline_orientation=True,
                det_limit_side_len=800
            )
            print("[INFO] PaddleOCR初始化成功")
            signal.alarm(0)  # 取消超時
        except TimeoutException:
            print("[ERROR] PaddleOCR初始化超時，可能需要更多内存")
            return [], 1.0
        except Exception as e:
            print(f"[ERROR] PaddleOCR初始化失敗: {e}")
            signal.alarm(0)  # 取消超時
            return [], 1.0
    
    # 讀取並壓縮圖片
    try:
        img, scale = safe_resize(image_path, max_side=800)
        arr = np.array(img)
    except Exception as e:
        print(f"[ERROR] 圖片處理失敗: {e}")
        return [], 1.0

    # 使用全局OCR實例進行識別
    try:
        signal.alarm(300)  # 5分鐘超時
        
        if hasattr(ocr_instance, "predict"):
            res = ocr_instance.predict(arr)   # PaddleOCR 3.0.0
        else:
            res = ocr_instance.ocr(arr, cls=False)  # Older fallback

        print(f"[INFO] OCR處理完成，檢測到 {len(res[0]) if res and res[0] else 0} 個文本框")

        signal.alarm(0)  # 取消超時

    except TimeoutException:
        print("[ERROR] OCR处理超时，图片可能太大或太复杂")
        return [], scale
    except Exception as e:
        print(f"[ERROR] OCR处理失败: {e}")
        signal.alarm(0)  # 取消超時
        return [], scale
    
    boxes = []
    if not res or not res[0]:
        return boxes, scale

    # 處理檢測結果並壓縮座標回原始尺寸
    for line in res[0]:
        if line is None:
            continue

    # PaddleOCR v3 典型輸出: [ box, (text, score) ]
        box = line[0]
        if len(line) > 1 and isinstance(line[1], tuple):
            text, score = line[1]
        else:
            text, score = "", 1.0

        if drop_rec:
            text, score = "", 1.0

    # 縮放回原始尺寸
        scaled_poly = []
        for point in box:
        # 檢查座標格式
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                print("[WARN] 跳過異常點:", point, "來自 box:", box)
                continue

            try:
                x, y = point[:2]   # 只取前兩個數，避免多維
                scaled_x = int(x / scale)
                scaled_y = int(y / scale)
                scaled_poly.append([scaled_x, scaled_y])
            except Exception as e:
                print("[ERROR] 無法處理點:", point, "錯誤原因:", e)

        boxes.append({"poly": scaled_poly, "text": text, "score": float(score)})
    
    return boxes, scale

# ---------- 可視化（標框） ----------
def visualize_boxes(image_path, boxes, save_path):
    # 使用原始圖片進行可視化
    try:
        img = Image.open(image_path).convert("RGB")
        # 如果圖片太大，先縮小再可視化
        w, h = img.size
        if max(w, h) > 2000:
            scale = 2000 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            # 同時縮放框坐標
            for box in boxes:
                for point in box["poly"]:
                    point[0] = int(point[0] * scale)
                    point[1] = int(point[1] * scale)
        
        draw = ImageDraw.Draw(img)
        for b in boxes:
            poly = b["poly"]
            # 繪製四邊形
            for i in range(4):
                start_point = tuple(poly[i])
                end_point = tuple(poly[(i + 1) % 4])
                draw.line([start_point, end_point], fill="red", width=3)
        img.save(save_path)
        print(f"[INFO] 可視化圖片已保存: {save_path}")
    except Exception as e:
        print(f"[ERROR] 可视化失敗: {e}")

# ---------- 直排右起排序 ----------
def sort_vertical_rtl(boxes):
    def x_center(b): return sum(p[0] for p in b["poly"]) / 4.0
    def y_top(b):    return min(p[1] for p in b["poly"])
    return sorted(boxes, key=lambda b: (-x_center(b), y_top(b)))

# ---------- 轉 PAGE-XML（最小可用版） ----------
def to_pagexml(image_path, boxes_sorted, save_xml, with_rec=False):
    try:
        img = Image.open(image_path)
        W, H = img.size
        
        NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
        ET.register_namespace('', NS)
        def E(tag, *args, **kwargs): return ET.Element(f"{{{NS}}}{tag}", *args, **kwargs)

        PcGts = E("PcGts")
        Page  = E("Page", imageFilename=os.path.basename(image_path),
                  imageWidth=str(W), imageHeight=str(H))
        PcGts.append(Page)
        region = E("TextRegion", id="r1")
        Page.append(region)

        for i, b in enumerate(boxes_sorted, 1):
            line = E("TextLine", id=f"l{i}")
            pts  = " ".join(f"{int(x)},{int(y)}" for (x, y) in b["poly"])
            line.append(E("Coords", points=pts))

            if with_rec and b.get("text"):
                te  = E("TextEquiv")
                uni = E("Unicode"); uni.text = b["text"]
                te.append(uni); line.append(te)

            region.append(line)

        tree = ET.ElementTree(PcGts)
        tree.write(save_xml, encoding="utf-8", xml_declaration=True)
        print(f"[INFO] PAGE-XML已保存: {save_xml}")
    except Exception as e:
        print(f"[ERROR] PAGE-XML生成失败: {e}")

# ---------- 主程式 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",     required=True, help="輸入圖片檔路徑")
    ap.add_argument("--outdir",    default="out", help="輸出資料夾")
    ap.add_argument("--lang",      default="ch",  help="語言代碼(ch=中文通用）")
    ap.add_argument("--to_pagexml", action="store_true", help="輸出 PAGE-XML")
    ap.add_argument("--with_rec",   action="store_true", help="在 PAGE-XML 中預填文字")
    args = ap.parse_args()

    try:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

        # drop_rec=True → 只分割；False → 分割+文字
        print("[INFO] 開始偵測文字框...")
        boxes, scale = detect_boxes(args.image, lang=args.lang, drop_rec=(not args.with_rec))
        print(f"[INFO] 偵測到 {len(boxes)} 個文字框")

        if not boxes:
            print("[WARNING] 未检测到任何文本框，程序退出")
            return

        json_path = os.path.join(args.outdir, Path(args.image).stem + ".det.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 偵測結果已輸出：{json_path}")

        vis_path = os.path.join(args.outdir, Path(args.image).stem + "_det_vis.jpg")
        visualize_boxes(args.image, boxes, vis_path)
        print(f"[OK] 偵測框可視化已輸出：{vis_path}")

        if args.to_pagexml:
            boxes_sorted = sort_vertical_rtl(boxes)
            xml_path = os.path.join(args.outdir, Path(args.image).stem + ".page.xml")
            to_pagexml(args.image, boxes_sorted, xml_path, with_rec=args.with_rec)
            print(f"[OK] PAGE-XML 已輸出：{xml_path}")
            print("    匯入 eScriptorium: Images → Import → Transcription (XML)，")
            print("    然後在 Segmentation 面板校正行框/基線，儲存後即成為你的 GT。")
            
    except Exception as e:
        print(f"[ERROR] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 確保取消所有的超時設置
        signal.alarm(0)

if __name__ == "__main__":
    main()