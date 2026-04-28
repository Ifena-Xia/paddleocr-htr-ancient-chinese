# 使用Sauvola二值化
# python3 paddle_single_sauvola.py --image 10_7eea5_default.jpg --outdir out --lang ch --to_pagexml --preprocess binarize_sauvola

import os, json, argparse, time, signal
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from paddleocr import PaddleOCR
import xml.etree.ElementTree as ET
import cv2
from skimage.filters import threshold_sauvola
from xml.dom import minidom

# 全局OCR實例，避免重複初始化
ocr_instance = None

# ---------- 超時處理 ----------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

# ---------- 基本圖片縮放 ----------
def safe_resize(image_path, max_side=1600): 
    img = Image.open(image_path)
    
    # 如果不是RGB，转换为RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 调整大小
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] 图片缩小为 {new_w}x{new_h} (缩放比例: {scale:.3f})")
    
    return img, scale

# Sauvola二值化函數
def binarize_sauvola_image(image_path, max_side=1600):
    """
    使用Sauvola算法進行二值化預處理
    """
    img = Image.open(image_path)
    w, h = img.size
    
    # 轉換為OpenCV格式
    cv_img = np.array(img.convert('RGB'))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Sauvola二值化 - 適合文檔圖像
    thresh = threshold_sauvola(gray, window_size=25)
    binary = (gray > thresh).astype(np.uint8) * 255
    
    # 轉回PIL格式
    img = Image.fromarray(binary)
    print(f"[INFO] 已應用Sauvola二值化預處理")

    # 調整大小
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] 圖片縮小為 {new_w}x{new_h} (縮放比例: {scale:.3f})")
    
    return img, scale

# ---------- 偵測文字框 ----------
def detect_boxes(image_path, lang='ch', drop_rec=True, preprocess_mode="none"):
    global ocr_instance
    
    if ocr_instance is None:
        print("[INFO] 初始化PaddleOCR...")
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)
            
            # OCR參數
            ocr_instance = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                cls=True,
                det_db_thresh=0.3,
                det_db_box_thresh=0.4,
                det_db_unclip_ratio=1.8,
                use_dilation=True,
                det_limit_side_len=1600,
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
    
    # 讀取圖片
    try:
        if preprocess_mode == "binarize_sauvola":
            img, scale = binarize_sauvola_image(image_path, max_side=1600)
        else:
            img, scale = safe_resize(image_path, max_side=1600)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img)
    except Exception as e:
        print(f"[ERROR] 圖片處理失敗: {e}")
        return [], 1.0

    # 使用OCR進行文字偵測
    try:
        signal.alarm(300)
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

    # 處理檢測結果
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
    
    return boxes, scale

# ---------- 可視化 ----------
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
        print(f"[INFO] 可视化图片已保存: {save_path}")
    except Exception as e:
        print(f"[ERROR] 可视化失败: {e}")

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
    """將XML元素轉換為美化的字符串"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# ---------- 轉 PAGE-XML（簡化版，但符合eScriptorium要求） ----------
def to_pagexml(image_path, boxes_sorted, save_xml, with_rec=False):
    try:
        img = Image.open(image_path)
        W, H = img.size
        
        # 創建根元素
        NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
        root = ET.Element("PcGts", {
            "xmlns": NS,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{NS} {NS}/pagecontent.xsd"
        })
        
        # 添加Metadata
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "Creator").text = "PaddleOCR HTR Pipeline"
        ET.SubElement(metadata, "Created").text = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # 添加Page
        page = ET.SubElement(root, "Page", {
            "imageFilename": os.path.basename(image_path),
            "imageWidth": str(W),
            "imageHeight": str(H)
        })
        
        # 添加文字區域
        region = ET.SubElement(page, "TextRegion", {"id": "r1"})
        ET.SubElement(region, "Coords", {"points": f"0,0 {W},0 {W},{H} 0,{H}"})
        
        # 添加文字行
        for i, box in enumerate(boxes_sorted, 1):
            if not box["poly"] or len(box["poly"]) != 4:
                continue
                
            line = ET.SubElement(region, "TextLine", {"id": f"l{i}"})
            pts = " ".join(f"{int(x)},{int(y)}" for (x, y) in box["poly"])
            ET.SubElement(line, "Coords", {"points": pts})

            # 加入 Baseline (vertical, top → bottom)
            poly = box["poly"]

            # 上邊中點（poly[0] 和 poly[1]）
            x_top = (poly[0][0] + poly[1][0]) // 2
            y_top = (poly[0][1] + poly[1][1]) // 2

            # 下邊中點（poly[2] 和 poly[3]）
            x_bottom = (poly[2][0] + poly[3][0]) // 2
            y_bottom = (poly[2][1] + poly[3][1]) // 2

            baseline_points = f"{x_top},{y_top} {x_bottom},{y_bottom}"
            ET.SubElement(line, "Baseline", {"points": baseline_points})
            
            if with_rec and box.get("text"):
                te = ET.SubElement(line, "TextEquiv")
                ET.SubElement(te, "Unicode").text = box["text"]
        
        # 寫入文件
        tree = ET.ElementTree(root)
        tree.write(save_xml, encoding="utf-8", xml_declaration=True)
        print(f"[INFO] PAGE-XML已保存: {save_xml}")
        
    except Exception as e:
        print(f"[ERROR] PAGE-XML生成失败: {e}")
        import traceback
        traceback.print_exc()

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="輸入圖片檔路徑")
    ap.add_argument("--outdir", default="out", help="輸出文件夾")
    ap.add_argument("--lang", default="ch", help="語言代碼")
    ap.add_argument("--to_pagexml", action="store_true", help="輸出 PAGE-XML")
    ap.add_argument("--with_rec", action="store_true", help="在 PAGE-XML 中預填文字")
    ap.add_argument("--max_side", type=int, default=1600, help="圖像最大邊長")
    ap.add_argument("--preprocess", 
                    choices=["none", "binarize_sauvola"], 
                    default="none", 
                    help="預處理模式: none(無), binarize_sauvola(Sauvola二值化)")
    args = ap.parse_args()

    try:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

        print("[INFO] 開始偵測文本框...")
        boxes, scale = detect_boxes(args.image, lang=args.lang, 
                                   drop_rec=(not args.with_rec), 
                                   preprocess_mode=args.preprocess)
        
        print(f"[INFO] 偵測到 {len(boxes)} 個文本框")

        if not boxes:
            print("[WARNING] 未檢測到任何的文本框，程序退出")
            return

        # 保存JSON結果
        json_path = os.path.join(args.outdir, Path(args.image).stem + ".det.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 偵測結果已輸出：{json_path}")

        # 可視化結果
        vis_path = os.path.join(args.outdir, Path(args.image).stem + "_det_vis.jpg")
        visualize_boxes(args.image, boxes, vis_path)
        print(f"[OK] 偵測框可視化已輸出：{vis_path}")

        # 生成PAGE XML
        if args.to_pagexml:
            boxes_sorted = sort_vertical_rtl(boxes)
            # 使用.xml擴展名（eScriptorium偏好）
            xml_path = os.path.join(args.outdir, Path(args.image).stem + ".xml")
            to_pagexml(args.image, boxes_sorted, xml_path, with_rec=args.with_rec)
            print(f"[OK] PAGE-XML 已輸出：{xml_path}")
            print("\n匯入 eScriptorium 步驟:")
            print("1. 在 eScriptorium 中創建或選擇專案")
            print("2. 點擊 Images → Import images")
            print("3. 上傳圖片和對應的 .xml 文件")
            print("4. 在 Segmentation 面板校正行框/基線")
            print("5. 儲存後即成為 Ground Truth")
            
    except Exception as e:
        print(f"[ERROR] 程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    main()