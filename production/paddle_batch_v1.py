# 用法：
#  單張圖片：python3 paddle_batch_v1.py --image page1.jpg --outdir out --lang ch --to_pagexml --with_rec
#  批量處理：python3 paddle_batch_v1.py --input_dir images --outdir out --lang ch --to_pagexml --with_rec

import os, json, argparse, time, signal, glob
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from paddleocr import PaddleOCR
import xml.etree.ElementTree as ET
from xml.dom import minidom

# 全局OCR實例，避免重複初始化
ocr_instance = None

# ---------- 超時處理 ----------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

# ---------- 放崩潰：過大圖片先縮邊 ----------
def safe_resize(image_path, max_side=1200):
    img = Image.open(image_path)
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] 图片缩小为 {new_w}x{new_h} (缩放比例: {scale:.3f})")
    return img, scale

# ---------- 偵測（可丢文字） ----------
def detect_boxes(image_path, lang='ch', drop_rec=True):
    """
    drop_rec=True  → 只要框（做 base segmentation 初稿）
    drop_rec=False → 框 + 文字（把文字也写进 PAGE-XML 当预填）
    回传: list[dict]，每个元素: {"poly": [[x1,y1]..[x4,y4]], "text": str, "score": float}
    """
    global ocr_instance
    
    # 初始化OCR實例（僅一次）
    if ocr_instance is None:
        print("[INFO] 初始化PaddleOCR...")
        try:
            # 設置超時處理
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 2分鐘超時
            
            # 使用最新的API參數，針對豎排文本
            ocr_instance = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                cls=True,
                det_db_thresh=0.3,
                det_db_box_thresh=0.4,
                det_db_p_ratio=1.8,
                use_dilation=True,
                det_limit_side_len=1200
            )
            print("[INFO] PaddleOCR初始化成功")
            signal.alarm(0)
        except TimeoutException:
            print("[ERROR] PaddleOCR初始化超時")
            return [], 1.0
        except Exception as e:
            print(f"[ERROR] PaddleOCR初始化失敗: {e}")
            signal.alarm(0)
            return [], 1.0
    
    # 讀取並壓縮圖片
    try:
        img, scale = safe_resize(image_path, max_side=1200)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img)
    except Exception as e:
        print(f"[ERROR] 圖片處理失敗: {e}")
        return [], 1.0

    # 使用全局OCR實例進行識別
    try:
        signal.alarm(300)  # 5分鍾超時
        res = ocr_instance.ocr(arr, cls=True)
        print(f"[INFO] OCR處理完成，檢測到 {len(res[0]) if res and res[0] else 0} 個文本框")
        signal.alarm(0)
    except TimeoutException:
        print("[ERROR] OCR處理超時")
        return [], scale
    except Exception as e:
        print(f"[ERROR] OCR處理失敗: {e}")
        signal.alarm(0)
        return [], scale
    
    boxes = []
    if not res or not res[0]:
        return boxes, scale

    # 處理檢測結果並壓縮座標回到原始尺寸
    for line in res[0]:
        if line is None:
            continue

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
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x, y = point[:2]
                scaled_x = int(x / scale)
                scaled_y = int(y / scale)
                scaled_poly.append([scaled_x, scaled_y])
            except Exception as e:
                print("[ERROR] 無法處理點:", point, "錯誤原因:", e)
                continue

        if len(scaled_poly) == 4:
            boxes.append({"poly": scaled_poly, "text": text, "score": float(score)})
        else:
            print(f"[WARN] 跳過異常文本，點數: {len(scaled_poly)}")
    
    return boxes, scale

# ---------- 可視化（標框） ----------
def visualize_boxes(image_path, boxes, save_path):
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > 2000:
            scale = 2000 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            for box in boxes:
                for point in box["poly"]:
                    point[0] = int(point[0] * scale)
                    point[1] = int(point[1] * scale)
        
        draw = ImageDraw.Draw(img)
        for b in boxes:
            poly = b["poly"]
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

# ---------- 美化XML輸出 ----------
def prettify(elem):
    """将XML元素转换为美化的字符串"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# ---------- Convert to PAGE-XML ----------
def to_pagexml(image_path, boxes_sorted, save_xml, with_rec=False):
    try:
        img = Image.open(image_path)
        W, H = img.size
        
        # Acquire the defualt name
        image_filename = os.path.basename(image_path)
        
        # Create root element
        NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
        root = ET.Element("PcGts", {
            "xmlns": NS,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{NS} {NS}/pagecontent.xsd"
        })
        
        # Add Metadata
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "Creator").text = "PaddleOCR HTR Pipeline"
        ET.SubElement(metadata, "Created").text = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Add Page - use the defualt name
        page = ET.SubElement(root, "Page", {
            "imageFilename": image_filename,  # Here is the change
            "imageWidth": str(W),
            "imageHeight": str(H)
        })
        
        # Add text region
        region = ET.SubElement(page, "TextRegion", {"id": "r1"})
        ET.SubElement(region, "Coords", {"points": f"0,0 {W},0 {W},{H} 0,{H}"})
        
        # Add text lines
        for i, box in enumerate(boxes_sorted, 1):
            if not box["poly"] or len(box["poly"]) != 4:
                continue
                
            line = ET.SubElement(region, "TextLine", {"id": f"l{i}"})
            pts = " ".join(f"{int(x)},{int(y)}" for (x, y) in box["poly"])
            ET.SubElement(line, "Coords", {"points": pts})

            # Add Baseline (vertical, top → bottom)
            poly = box["poly"]

            # Midpoint of the top edge (poly[0] and poly[1])
            x_top = (poly[0][0] + poly[1][0]) // 2
            y_top = (poly[0][1] + poly[1][1]) // 2

            # Midpoint of the bottom edge (poly[2] and poly[3])
            x_bottom = (poly[2][0] + poly[3][0]) // 2
            y_bottom = (poly[2][1] + poly[3][1]) // 2

            baseline_points = f"{x_top},{y_top} {x_bottom},{y_bottom}"
            ET.SubElement(line, "Baseline", {"points": baseline_points})
            
            if with_rec and box.get("text"):
                te = ET.SubElement(line, "TextEquiv")
                ET.SubElement(te, "Unicode").text = box["text"]
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(save_xml, encoding="utf-8", xml_declaration=True)
        print(f"[INFO] PAGE-XML saved: {save_xml}")
        
    except Exception as e:
        print(f"[ERROR] PAGE-XML generation failed: {e}")
        import traceback
        traceback.print_exc()

# ---------- 處理單張圖片 ----------
def process_single_image(image_path, args):
    """處理單張圖片並生成所有輸出文件"""
    print(f"\n[INFO] 开始处理图片: {image_path}")
    
    try:
        boxes, scale = detect_boxes(image_path, lang=args.lang, drop_rec=(not args.with_rec))
        print(f"[INFO] 檢測到 {len(boxes)} 個文本框")

        if not boxes:
            print(f"[WARNING] 圖片 {image_path} 未檢測到任何文本框，跳過")
            return

        # 獲取圖片的基本文件名（不含擴展名）
        stem = Path(image_path).stem
        
        # 保存JSON結果
        json_path = os.path.join(args.outdir, stem + ".det.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 檢測結果已輸出：{json_path}")

        # 可視化結果
        vis_path = os.path.join(args.outdir, stem + "_det_vis.jpg")
        visualize_boxes(image_path, boxes, vis_path)
        print(f"[OK] 檢測框可視化已輸出：{vis_path}")

        # 生成PAGE XML
        if args.to_pagexml:
            boxes_sorted = sort_vertical_rtl(boxes)
            xml_path = os.path.join(args.outdir, stem + ".xml")
            to_pagexml(image_path, boxes_sorted, xml_path, with_rec=args.with_rec)
            print(f"[OK] PAGE-XML 已輸出：{xml_path}")
            
    except Exception as e:
        print(f"[ERROR] 處理圖片 {image_path} 時失敗: {e}")
        import traceback
        traceback.print_exc()

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",     help="輸入單張圖片文件路徑")
    ap.add_argument("--input_dir", help="输入圖片文件夾路徑（批量處理）")
    ap.add_argument("--outdir",    default="out", help="輸出文件夾")
    ap.add_argument("--lang",      default="ch",  help="語言代碼(ch=中文通用）")
    ap.add_argument("--to_pagexml", action="store_true", help="輸出 PAGE-XML")
    ap.add_argument("--with_rec",   action="store_true", help="在 PAGE-XML 中預填文字")
    ap.add_argument("--max_side",   type=int, default=1200, help="圖像最大邊長（默認1200）")
    args = ap.parse_args()

    # 檢查參數
    if not args.image and not args.input_dir:
        print("[ERROR] 必須指定 --image 或 --input_dir 參數")
        return
    
    if args.image and args.input_dir:
        print("[ERROR] 不能同時指定 --image 和 --input_dir 參數")
        return

    try:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

        # 處理單張圖片
        if args.image:
            if not os.path.exists(args.image):
                print(f"[ERROR] 圖片文件不存在: {args.image}")
                return
            process_single_image(args.image, args)
        
        # Batch process an image folder
        elif args.input_dir:
            if not os.path.exists(args.input_dir):
                print(f"[ERROR] Input folder does not exist: {args.input_dir}")
                return
            
            # Supported image formats
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
                image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
            
            if not image_files:
                print(f"[WARNING] No image files found in folder {args.input_dir}")
                return
            
            print(f"[INFO] Found {len(image_files)} images. Starting batch processing...")
            
            # Sort by filename to ensure consistent order
            image_files.sort()
            
            for i, image_path in enumerate(image_files, 1):
                print(f"\n{'='*50}")
                print(f"Processing progress: {i}/{len(image_files)}")
                print(f"{'='*50}")
                process_single_image(image_path, args)
            
            print(f"\n[INFO] Batch processing complete! Total processed: {len(image_files)} images")
            print("\n導入 eScriptorium 步驟:")
            print("1. 在 eScriptorium 中創建或選擇項目")
            print("2. 點擊 Images → Import images")
            print("3. 上傳所有圖片和對應的 .xml 文件")
            print("4. 在 Segmentation 面板校正行框/基線")
            print("5. 保存後即成為 Ground Truth")
            
    except Exception as e:
        print(f"[ERROR] 程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    main()