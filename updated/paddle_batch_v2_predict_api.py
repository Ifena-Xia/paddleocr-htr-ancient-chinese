# 用法：
#  單張圖片：python3 paddle_batch_v2_predict_api.py --image 圖片名稱 --outdir out --lang ch --to_pagexml
#  批量處理：python3 paddle_batch_v2_predict_api.py --input_dir 文件夾名稱 --outdir out --lang ch --to_pagexml

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
        print(f"[INFO] 圖片縮小為 {new_w}x{new_h} (縮放比例: {scale:.3f})")
    return img, scale

# ---------- 偵測（可丢文字） ----------
def detect_boxes(image_path, lang='ch', drop_rec=True):
    """
    drop_rec=True  → 只要框（做 base segmentation 初稿）
    drop_rec=False → 框 + 文字（把文字也寫進 PAGE-XML 當預填）
    回傳: list[dict]，每個元素: {"poly": [[x1,y1]..[x4,y4]], "text": str, "score": float}
    """
    global ocr_instance
    
    # 初始化OCR實例（僅一次）
    if ocr_instance is None:
        print("[INFO] 初始化PaddleOCR...")
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)
            
            # 使用新的参数名称
            ocr_instance = PaddleOCR(
                lang=lang,
                # 關掉不需要的子模組，避免奇怪的自動旋轉、糾正
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True, 

                # 對應以前的 det_db_* 參數
                text_det_limit_side_len=1200,    # 原 det_limit_side_len
                text_det_thresh=0.3,             # 原 det_db_thresh
                text_det_box_thresh=0.4,         # 原 det_db_box_thresh
                text_det_unclip_ratio=1.6,       # 原 det_db_p_ratio / det_db_unclip_ratio
            )

            print("[INFO] PaddleOCR 初始化成功")
            signal.alarm(0)
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
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    except Exception as e:
        print(f"[ERROR] 圖片處理失敗: {e}")
        return [], 1.0

    # 使用全局OCR實例進行識別
    try:
        signal.alarm(300)
        res = ocr_instance.predict(arr)
        signal.alarm(0)
    except Exception as e:
        print(f"[ERROR] OCR處理失敗: {e}")
        signal.alarm(0)
        return [], scale
    
    boxes = []
    
    # 檢查OCR結果結構
    if res is None or len(res) == 0:
        print("[WARN] OCR返回結果為空")
        return boxes, scale
    
    # 獲取第一頁的OCR結果（OCRResult對象）
    ocr_result = res[0]
    
    print(f"[DEBUG] OCR結果對象類型: {type(ocr_result)}")
    
    # 從 json 屬性中提取 OCR 結果
    try:
        if hasattr(ocr_result, 'json'):
            json_data = ocr_result.json
            print("[DEBUG] ====== 分析 json 屬性結構 ======")
            
            # 檢查 json 結構
            if isinstance(json_data, dict) and 'res' in json_data:
                res_data = json_data['res']
                
                # 直接使用已知的鍵名提取數據
                dt_polys = res_data.get('dt_polys', [])
                rec_texts = res_data.get('rec_texts', [])
                rec_scores = res_data.get('rec_scores', [])
                
                print(f"[DEBUG] 找到 dt_polys: {len(dt_polys)} 個")
                print(f"[DEBUG] 找到 rec_texts: {len(rec_texts)} 個")
                print(f"[DEBUG] 找到 rec_scores: {len(rec_scores)} 個")
                
                # 處理每個文本框
                for i, poly in enumerate(dt_polys):
                    try:
                        # 獲取對應的文字和置信度
                        text = rec_texts[i] if i < len(rec_texts) else ""
                        score = rec_scores[i] if i < len(rec_scores) else 1.0
                        
                        # 如果只需要框，清空文字
                        if drop_rec:
                            text = ""
                        
                        # 處理多邊形座標
                        scaled_poly = []
                        for point in poly:
                            if len(point) >= 2:
                                x, y = float(point[0]), float(point[1])
                                scaled_x = int(x / scale) if scale != 1.0 else int(x)
                                scaled_y = int(y / scale) if scale != 1.0 else int(y)
                                scaled_poly.append([scaled_x, scaled_y])
                        
                        if len(scaled_poly) >= 4:
                            boxes.append({
                                "poly": scaled_poly, 
                                "text": str(text), 
                                "score": float(score)
                            })
                            
                            # 打印前几个框的详细信息
                            if len(boxes) <= 3:
                                print(f"[DEBUG] 成功處理文本框 {i}: 文字='{text}', 置信度={score:.3f}")
                        
                    except Exception as e:
                        print(f"[WARN] 處理文本框 {i} 時出錯: {e}")
                        continue
            else:
                print("[WARN] 無法從 json 屬性中獲取 res 數據")
        else:
            print("[WARN] OCRResult 對象沒有 json 屬性")
            
    except Exception as e:
        print(f"[ERROR] 解析 OCR 結果時出錯: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[INFO] 成功提取 {len(boxes)} 個有效文本框")
    return boxes, scale

# ---------- 可視化（標框） ----------
def visualize_boxes(image_path, boxes, save_path):
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        
        # 創建副本用於繪製，避免修改原始boxes
        draw_boxes = []
        for box in boxes:
            draw_boxes.append({
                "poly": [point[:] for point in box["poly"]],  
                "text": box["text"],
                "score": box["score"]
            })
        
        # 如果圖片太大，縮小用於可視化
        draw_img = img
        if max(w, h) > 2000:
            scale = 2000 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            draw_img = img.resize((new_w, new_h), Image.LANCZOS)
            # 按同樣比例縮小坐標
            for box in draw_boxes:
                for point in box["poly"]:
                    point[0] = int(point[0] * scale)
                    point[1] = int(point[1] * scale)
        
        draw = ImageDraw.Draw(draw_img)
        for b in draw_boxes:
            poly = b["poly"]
            # 繪製邊框
            for i in range(4):
                start_point = tuple(poly[i])
                end_point = tuple(poly[(i + 1) % 4])
                draw.line([start_point, end_point], fill="red", width=3)
            # 可選：在框中心添加序號
            center_x = sum(p[0] for p in poly) // 4
            center_y = sum(p[1] for p in poly) // 4
            draw.text((center_x, center_y), str(boxes.index(b)), fill="blue")
        
        draw_img.save(save_path)
        print(f"[INFO] 可視化圖片已保存: {save_path}")
    except Exception as e:
        print(f"[ERROR] 可視化失敗: {e}")
        import traceback
        traceback.print_exc()

# ---------- 直排右起排序 ----------
def sort_vertical_rtl(boxes):
    if not boxes:
        return boxes
        
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

# ---------- Convert to PAGE-XML ----------
def to_pagexml(image_path, boxes_sorted, save_xml, with_rec=False):
    try:
        img = Image.open(image_path)
        W, H = img.size
        
        # 獲取文件名
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
        
        # Add Page
        page = ET.SubElement(root, "Page", {
            "imageFilename": image_filename,
            "imageWidth": str(W),
            "imageHeight": str(H)
        })
        
        # Add text region
        region = ET.SubElement(page, "TextRegion", {"id": "r1"})
        ET.SubElement(region, "Coords", {"points": f"0,0 {W},0 {W},{H} 0,{H}"})
        
        # Add text lines
        for i, box in enumerate(boxes_sorted, 1):
            if not box["poly"] or len(box["poly"]) != 4:
                print(f"[WARN] 跳過無效的文本框 {i}")
                continue
                
            line = ET.SubElement(region, "TextLine", {"id": f"l{i}"})
            pts = " ".join(f"{int(x)},{int(y)}" for (x, y) in box["poly"])
            ET.SubElement(line, "Coords", {"points": pts})

            # Add Baseline (vertical, top → bottom)
            poly = box["poly"]
            x_top = (poly[0][0] + poly[1][0]) // 2
            y_top = (poly[0][1] + poly[1][1]) // 2
            x_bottom = (poly[2][0] + poly[3][0]) // 2
            y_bottom = (poly[2][1] + poly[3][1]) // 2

            baseline_points = f"{x_top},{y_top} {x_bottom},{y_bottom}"
            ET.SubElement(line, "Baseline", {"points": baseline_points})
            
            if with_rec and box.get("text") and box["text"].strip():
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
        print(f"[INFO] 检测完成，获得 {len(boxes)} 个文本框")

        if not boxes:
            print(f"[WARNING] 图片 {image_path} 未检测到任何文本框，跳过")
            return

        # 獲取圖片的基本文件名（不含擴展名）
        stem = Path(image_path).stem
        
        # 保存JSON結果
        json_path = os.path.join(args.outdir, stem + ".det.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 检测结果已输出：{json_path}")

        # 可視化結果
        vis_path = os.path.join(args.outdir, stem + "_det_vis.jpg")
        visualize_boxes(image_path, boxes, vis_path)
        print(f"[OK] 检测框可视化已输出：{vis_path}")

        # 生成PAGE XML
        if args.to_pagexml:
            boxes_sorted = sort_vertical_rtl(boxes)
            xml_path = os.path.join(args.outdir, stem + ".xml")
            to_pagexml(image_path, boxes_sorted, xml_path, with_rec=args.with_rec)
            print(f"[OK] PAGE-XML 已输出：{xml_path}")
            
    except Exception as e:
        print(f"[ERROR] 处理图片 {image_path} 时失败: {e}")
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
        
        # 批量處理圖片文件夾
        elif args.input_dir:
            if not os.path.exists(args.input_dir):
                print(f"[ERROR] 輸入文件夾不存在: {args.input_dir}")
                return
            
            # 支持的圖片格式
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
                image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
            
            if not image_files:
                print(f"[WARNING] 在文件夾 {args.input_dir} 中未找到圖片文件")
                return
            
            print(f"[INFO] 找到 {len(image_files)} 張圖片，開始批量處理...")
            
            # 按文件名排序以確保順序一致
            image_files.sort()
            
            for i, image_path in enumerate(image_files, 1):
                print(f"\n{'='*50}")
                print(f"處理進度: {i}/{len(image_files)}")
                print(f"{'='*50}")
                process_single_image(image_path, args)
            
            print(f"\n[INFO] 批量處理完成! 總共處理了 {len(image_files)} 張圖片")
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
