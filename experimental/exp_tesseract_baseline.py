# 用法：
#  只做分割：python3 exp_tesseract_baseline.py --image Page_3.png --outdir out --lang  chi_tra_vert  --to_pagexml
#  分割+文字：python3 exp_tesseract_baseline.py --image Page_3.png --outdir out --lang  chi_tra_vert  --to_pagexml --with_rec

import os
import json
import cv2
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET

try:
    import pytesseract
except ImportError:
    print("請先安裝 pytesseract： pip install pytesseract")
    exit(1)

# ---------- 防崩潰：過大圖片先縮邊 ----------
def safe_resize(image_path, max_side=2000):
    img = Image.open(image_path)
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"[INFO] 圖片縮小為 {img.size[0]}x{img.size[1]} (縮放比例: {scale:.3f})")
    return img, scale

# ---------- 使用 Tesseract 偵測文字框 ----------
def detect_boxes(image_path, lang='chi_tra_vert', drop_rec=True):
    # 讀取並壓縮圖片
    img, scale = safe_resize(image_path)
    gray = np.array(img.convert('L'))  # 直接灰階
    # 加入二值化處理（Otsu自動門檻）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, lang=lang)
    
    boxes = []
    for i in range(len(data['text'])):
        # 只處理置信度高的結果
        try:
            conf = int(data['conf'][i])
        except:
            continue
            
        if conf < 30:
            continue

        # 獲取邊界框座標
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        # 跳過無效的邊界框
        if w <= 0 or h <= 0:
            continue
            
        # 縮放回原始尺寸
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)
        
        # 創建多邊形座標
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        
        # 獲取文本
        text = data['text'][i].strip()
        if not text:  # 跳过空文本
            continue

        # 根據 drop_rec 參數決定是否保留識別文本
        if drop_rec:
            text = " "

        boxes.append({
            "poly": poly,
            "text": text,
            "score": conf / 100.0
        })
    
    return boxes, scale

# ---------- 可視化（標框） ----------
def visualize_boxes(image_path, boxes, save_path):
    try:
        img = Image.open(image_path).convert("RGB")
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
        print(f"[ERROR] 可視化失敗: {e}")

# ---------- 直排右起排序 ----------
def sort_vertical_rtl(boxes):
    def x_center(b): 
        if not b["poly"]:
            return 0
        return sum(p[0] for p in b["poly"]) / 4.0
        
    def y_top(b):    
        if not b["poly"]:
            return 0
        return min(p[1] for p in b["poly"])
        
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
            # 确保边界框有效
            if not b["poly"] or len(b["poly"]) != 4:
                continue
                
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
    ap.add_argument("--lang",      default="chi_tra_vert",  help="語言代碼（chi_tra_vert=中文直排, chi_tra=中文橫排）")
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

        stem = Path(args.image).stem

        # 輸出 JSON
        json_path = os.path.join(args.outdir, f"{stem}.det.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 偵測結果已輸出：{json_path}")

        # 視覺化
        vis_path = os.path.join(args.outdir, f"{stem}_det_vis.jpg")
        visualize_boxes(args.image, boxes, vis_path)

        # 轉 PAGE-XML
        if args.to_pagexml:
            boxes_sorted = sort_vertical_rtl(boxes)
            xml_path = os.path.join(args.outdir, f"{stem}.page.xml")
            to_pagexml(args.image, boxes_sorted, xml_path, with_rec=args.with_rec)
        
        # 自動標識模型名
        model_tag = args.lang.strip().replace(" ", "_")
        stem = f"{Path(args.image).stem}_{model_tag}"           
            
    except Exception as e:
        print(f"[ERROR] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()