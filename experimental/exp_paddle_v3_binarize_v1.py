# 嘗試自適應二值化
# python3 exp_paddle_v3_binarize_v1.py --image Page_9.png --outdir out --lang ch --to_pagexml --preprocess binarize_adaptive
# 嘗試Sauvola二值化（特別適合文檔）
# python3 exp_paddle_v3_binarize_v1.py --image Page_9.png --outdir out --lang ch --to_pagexml --preprocess binarize_sauvola
# 啟用後處理過濾
# python3 exp_paddle_v3_binarize_v1.py --image Page_9.png --outdir out --lang ch --to_pagexml --preprocess binarize_adaptive --filter_boxes

import os, json, argparse, time, signal
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
from paddleocr import PaddleOCR
import xml.etree.ElementTree as ET
import cv2
from skimage import exposure
from skimage.filters import threshold_sauvola  # 添加這行導入

# 全局OCR實例，避免重複初始化
ocr_instance = None

# ---------- 超時處理 ----------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

# ---------- 縮小 + 增加對比度 ----------
def safe_resize(image_path, max_side=1600, preprocess=False): 
    img = Image.open(image_path)
    
    # 預處理 - 增強對比度 轉換為灰度圖
    if preprocess:
        # 轉換為OpenCV格式
        cv_img = np.array(img.convert('RGB'))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # 轉換為灰度图
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 計算圖像的整體對比度
        contrast = np.std(gray)

        # 自适应调整CLAHE参数
        if contrast < 40:  # 低对比度图像
            clip_limit = 3.0
            tile_size = 12
            print(f"[INFO] 低对比度图像 ({contrast:.1f})，使用强预处理")
        elif contrast > 80:  # 高对比度图像
            clip_limit = 1.0
            tile_size = 16
            print(f"[INFO] 高对比度图像 ({contrast:.1f})，使用弱预处理")
        else:  # 中等对比度图像
            clip_limit = 2.0
            tile_size = 8
            print(f"[INFO] 中等对比度图像 ({contrast:.1f})，使用中等预处理")
        
        # 使用自适应CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(gray)
        
        # 转回PIL格式
        img = Image.fromarray(enhanced)
        print(f"[INFO] 已应用自适应图像预处理 (clipLimit={clip_limit}, tileSize={tile_size})")
    else:
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

# 選擇性預處理函數
def selective_preprocess(image_path, max_side=1600):
    """
    選擇性預處理：只對低對比度的區域應用
    """
    img = Image.open(image_path)
    w, h = img.size
    
    # 轉換為OpenCV格式
    cv_img = np.array(img.convert('RGB'))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # 計算局部對比度（標準差濾波器）
    from scipy.ndimage import generic_filter
    def std_filter(window):
        return np.std(window)
    
    # 使用較小的窗口計算局部對比度
    local_contrast = generic_filter(gray, std_filter, size=15)
    
    # 創建mask：只對低對比度區域預處理
    mask = local_contrast < 30  # 對比度閾值
    
    # 對低對比度區域應用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 合併結果：低對比度區域使用增強後的圖像，高對比度區域使用原圖
    result = np.where(mask, enhanced, gray)
    
    # 轉回PIL格式
    img = Image.fromarray(result)
    print("[INFO] 已應用選擇性預處理：只增強低對比度區域")

    # 調整大小
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] 圖片縮小為 {new_w}x{new_h} (縮放比例: {scale:.3f})")
    
    return img, scale

# 增加二值化
def binarize_image(image_path, max_side=1600, method='adaptive'):
    """
    二值化預處理
    method: 'adaptive' (自適應), 'otsu' (大津算法), 'sauvola' (Sauvola算法)
    """
    img = Image.open(image_path)
    w, h = img.size
    
    # 轉換為OpenCV格式
    cv_img = np.array(img.convert('RGB'))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    if method == 'adaptive':
        # 自適應二值化 - 適合光照不均的圖像
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    elif method == 'otsu':
        # Otsu二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'sauvola':
        # Sauvola二值化 - 適合文檔圖像
        thresh = threshold_sauvola(gray, window_size=25)
        binary = (gray > thresh).astype(np.uint8) * 255
    else:
        binary = gray
    
    # 可選的形態學操作來增強文字
    kernel = np.ones((1,1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 轉回PIL格式
    img = Image.fromarray(binary)
    print(f"[INFO] 已應用二值化預處理 (方法: {method})")

    # 調整大小
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] 圖片縮小為 {new_w}x{new_h} (縮放比例: {scale:.3f})")
    
    return img, scale

# ---------- 侦测（可丢文字）+ 改進參數配置 ----------
def detect_boxes(image_path, lang='ch', drop_rec=True, preprocess_mode="none"):
    global ocr_instance
    
    if ocr_instance is None:
        print("[INFO] 初始化PaddleOCR...")
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)
            
            # 改進的OCR參數 - 專門針對小文字和複雜背景
            ocr_instance = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                cls=True,
                # 降低檢測閾值以檢測更多文字
                det_db_thresh=0.2,  # 進一步降低
                det_db_box_thresh=0.3,  # 進一步降低
                det_db_unclip_ratio=2.0,  # 增加擴展比例
                use_dilation=True,
                # 專門針對小文字的參數
                det_limit_side_len=1600,  # 增加檢測尺寸限制
                det_db_score_mode='fast',  # 使用快速評分模式
                # 增加模型對小文字的敏感度
                rec_image_shape='3, 48, 320',  # 調整識別圖像形狀
            )
            print("[INFO] PaddleOCR初始化成功（小文字優化版）")
            signal.alarm(0)
        except TimeoutException:
            print("[ERROR] PaddleOCR初始化超時")
            return [], 1.0
        except Exception as e:
            print(f"[ERROR] PaddleOCR初始化失敗: {e}")
            signal.alarm(0)
            return [], 1.0
    
    # 讀取圖片 - 擴展預處理選項
    try:
        if preprocess_mode == "global":
            img, scale = safe_resize(image_path, max_side=1600, preprocess=True)
        elif preprocess_mode == "selective":
            img, scale = selective_preprocess(image_path, max_side=1600)
        elif preprocess_mode == "binarize_adaptive":
            img, scale = binarize_image(image_path, max_side=1600, method='adaptive')
        elif preprocess_mode == "binarize_otsu":
            img, scale = binarize_image(image_path, max_side=1600, method='otsu')
        elif preprocess_mode == "binarize_sauvola":
            img, scale = binarize_image(image_path, max_side=1600, method='sauvola')
        else:
            img, scale = safe_resize(image_path, max_side=1600, preprocess=False)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img)
    except Exception as e:
        print(f"[ERROR] 圖片處理失敗: {e}")
        return [], 1.0

    # 全局OCR實例進行識別
    try:
        signal.alarm(300)  # 5分鐘超時
        
        # API
        res = ocr_instance.ocr(arr, cls=True)  # 確保啟用方向分類

        print(f"[INFO] OCR處理完成，檢測到 {len(res[0]) if res and res[0] else 0} 個文本框")

        signal.alarm(0)  # 取消超時

    except TimeoutException:
        print("[ERROR] OCR處理超時，圖片太大或者太複雜")
        return [], scale
    except Exception as e:
        print(f"[ERROR] OCR處理失敗: {e}")
        signal.alarm(0)  # 取消超時
        return [], scale
    
    boxes = []
    if not res or not res[0]:
        return boxes, scale

    # 處理檢測結果並轉換坐標為原始尺寸
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

        # 回到原始尺寸
        scaled_poly = []
        for point in box:
            # 檢查坐標格式
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                print("[WARN] 跳過異常點:", point, "来自 box:", box)
                continue

            try:
                x, y = point[:2]   # 避免多維
                scaled_x = int(x / scale)
                scaled_y = int(y / scale)
                scaled_poly.append([scaled_x, scaled_y])
            except Exception as e:
                print("[ERROR] 無法處理點:", point, "錯誤原因:", e)
                continue

        # 確保多邊形有4個點
        if len(scaled_poly) == 4:
            boxes.append({"poly": scaled_poly, "text": text, "score": float(score)})
        else:
            print(f"[WARN] 跳过异常文本框，点数: {len(scaled_poly)}")
    
    return boxes, scale

# 過濾掉可能不是文字的區域
def filter_boxes(boxes, min_area=50, max_aspect_ratio=10, min_text_confidence=0.1):
    """
    過濾檢測結果，移除不符合條件的框
    """
    filtered_boxes = []
    
    for box in boxes:
        if not box["poly"] or len(box["poly"]) != 4:
            continue
            
        # 計算框的面積
        poly = np.array(box["poly"])
        area = cv2.contourArea(poly)
        
        # 計算寬高比
        x_coords = [p[0] for p in poly]
        y_coords = [p[1] for p in poly]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        if width == 0 or height == 0:
            continue
            
        aspect_ratio = max(width/height, height/width)
        
        # 過濾條件
        if (area >= min_area and 
            aspect_ratio <= max_aspect_ratio and 
            box.get("score", 1.0) >= min_text_confidence):
            filtered_boxes.append(box)
        else:
            print(f"[DEBUG] 過濾掉框: 面積={area}, 寬高比={aspect_ratio:.2f}")
    
    print(f"[INFO] 過濾後保留 {len(filtered_boxes)}/{len(boxes)} 個文本框")
    return filtered_boxes

# ---------- 可視化（標框） ----------
def visualize_boxes(image_path, boxes, save_path):
    # 使用原始圖片進行可視化
    try:
        img = Image.open(image_path).convert("RGB")
        # 圖片太大，先壓縮再可視化
        w, h = img.size
        if max(w, h) > 2000:
            scale = 2000 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            # 同時縮放坐標框
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
            # 確保邊界框有效
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
                    choices=["none", "global", "selective", 
                            "binarize_adaptive", "binarize_otsu", "binarize_sauvola"], 
                    default="none", 
                    help="預處理模式")
    ap.add_argument("--filter_boxes", action="store_true", help="啟用後處理過濾")
    args = ap.parse_args()

    try:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

        # drop_rec=True → 只分割；False → 分割+文字
        print("[INFO] 開始偵測文本框...")
        boxes, scale = detect_boxes(args.image, lang=args.lang, 
                                   drop_rec=(not args.with_rec), 
                                   preprocess_mode=args.preprocess)
        
        if args.filter_boxes:
            boxes = filter_boxes(boxes)
            
        print(f"[INFO] 偵測到 {len(boxes)} 個文本框")

        if not boxes:
            print("[WARNING] 未檢測到任何的文本框，程序退出")
            return

        json_path = os.path.join(args.outdir, Path(args.image).stem + ".det.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 偵測結果已輸出：{json_path}")

        vis_path = os.path.join(args.outdir, Path(args.image).stem + "_det_vis.jpg")
        visualize_boxes(args.image, boxes, vis_path)
        print(f"[OK]偵測框可視化已輸出：{vis_path}")

        if args.to_pagexml:
            boxes_sorted = sort_vertical_rtl(boxes)
            xml_path = os.path.join(args.outdir, Path(args.image).stem + ".page.xml")
            to_pagexml(args.image, boxes_sorted, xml_path, with_rec=args.with_rec)
            print(f"[OK] PAGE-XML 已输出：{xml_path}")
            print("    匯入 eScriptorium: Images → Import → Transcription (XML)，")
            print("    然後在 Segmentation 面板校正行框/基线，儲存後即成為你的 GT。")
            
    except Exception as e:
        print(f"[ERROR] 程序執行失敗: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 確保取消所有的超時設置
        signal.alarm(0)

if __name__ == "__main__":
    main()