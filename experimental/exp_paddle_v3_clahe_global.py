# 用法：
#  只做分割：python3 exp_paddle_v3_clahe_global.py --image 圖片名稱 --outdir out --lang ch --to_pagexml
#  應用於處理（增強對比度）：python3 exp_paddle_v3_clahe_global.py --image 圖片名稱 --outdir out --lang ch --to_pagexml --preprocess
#  分割+文字：python3 exp_paddle_v3_clahe_global.py --image 圖片名稱 --outdir out --lang ch --to_pagexml --with_rec

import os, json, argparse, time, signal
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
from paddleocr import PaddleOCR
import xml.etree.ElementTree as ET
import cv2

# 全局OCR实例，避免重复初始化
ocr_instance = None

# ---------- 超时处理 ----------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

# ---------- 縮小 + 增加對比度 ----------
def safe_resize(image_path, max_side=1200, preprocess=False):  # 增加最大尺寸到1200
    img = Image.open(image_path)
    
    # Image preprocessing (Augmenting contrast and converting to gray scale)
    if preprocess:
        # Converting to OpenCV format
        cv_img = np.array(img.convert('RGB'))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # Converting to gray scale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Converting back to PIL format
        img = Image.fromarray(enhanced)
        print("[INFO] Image preprocessing has been conducted")
    else:
        # Converting to RGB if not already in RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
    
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[INFO] 图片缩小为 {new_w}x{new_h} (缩放比例: {scale:.3f})")
    return img, scale

# ---------- 侦测（可丢文字） ----------
def detect_boxes(image_path, lang='ch', drop_rec=True, preprocess=False):
    """
    drop_rec=True  → 只要框（做 base segmentation 初稿）
    drop_rec=False → 框 + 文字（把文字也写进 PAGE-XML 当预填）
    回传: list[dict]，每个元素: {"poly": [[x1,y1]..[x4,y4]], "text": str, "score": float}
    """
    global ocr_instance
    
    # 初始化OCR实例（仅一次）
    if ocr_instance is None:
        print("[INFO] 初始化PaddleOCR...")
        try:
            # 设置超时处理
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 2分钟超时
            
            # 使用最新的API参数，特别针对竖排文本优化
            ocr_instance = PaddleOCR(
                lang=lang,
                use_angle_cls=True,  # 启用方向分类器
                cls=True,  # 启用文字方向分类
                det_db_thresh=0.3,  # 降低检测阈值
                det_db_box_thresh=0.4,  # 降低框阈值
                det_db_unclip_ratio=1.6,  # 增加扩展比例
                use_dilation=True,  # 使用膨胀来分离接近的文本行
                det_limit_side_len=1200  # 增加检测限制
            )
            print("[INFO] PaddleOCR初始化成功")
            signal.alarm(0)  # 取消超时
        except TimeoutException:
            print("[ERROR] PaddleOCR初始化超时，可能需要更多内存")
            return [], 1.0
        except Exception as e:
            print(f"[ERROR] PaddleOCR初始化失败: {e}")
            signal.alarm(0)  # 取消超时
            return [], 1.0
    
    # 读取并压缩图片
    try:
        img, scale = safe_resize(image_path, max_side=1200, preprocess=preprocess)  # 增加最大尺寸
        # 确保图像是三通道的
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img)
    except Exception as e:
        print(f"[ERROR] 图片处理失败: {e}")
        return [], 1.0

    # 使用全局OCR实例进行识别
    try:
        signal.alarm(300)  # 5分钟超时
        
        # 使用新版API
        res = ocr_instance.ocr(arr, cls=True)  # 确保启用方向分类

        print(f"[INFO] OCR处理完成，检测到 {len(res[0]) if res and res[0] else 0} 个文本框")

        signal.alarm(0)  # 取消超时

    except TimeoutException:
        print("[ERROR] OCR处理超时，图片可能太大或太复杂")
        return [], scale
    except Exception as e:
        print(f"[ERROR] OCR处理失败: {e}")
        signal.alarm(0)  # 取消超时
        return [], scale
    
    boxes = []
    if not res or not res[0]:
        return boxes, scale

    # 处理检测结果并压缩坐标回原始尺寸
    for line in res[0]:
        if line is None:
            continue

        # PaddleOCR v3 典型输出: [ box, (text, score) ]
        box = line[0]
        if len(line) > 1 and isinstance(line[1], tuple):
            text, score = line[1]
        else:
            text, score = "", 1.0

        if drop_rec:
            text, score = "", 1.0

        # 缩放回原始尺寸
        scaled_poly = []
        for point in box:
            # 检查坐标格式
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                print("[WARN] 跳过异常点:", point, "来自 box:", box)
                continue

            try:
                x, y = point[:2]   # 只取前两个数，避免多维
                scaled_x = int(x / scale)
                scaled_y = int(y / scale)
                scaled_poly.append([scaled_x, scaled_y])
            except Exception as e:
                print("[ERROR] 无法处理点:", point, "错误原因:", e)
                continue

        # 确保多边形有4个点
        if len(scaled_poly) == 4:
            boxes.append({"poly": scaled_poly, "text": text, "score": float(score)})
        else:
            print(f"[WARN] 跳过异常文本框，点数: {len(scaled_poly)}")
    
    return boxes, scale

# ---------- 可视化（标框） ----------
def visualize_boxes(image_path, boxes, save_path):
    # 使用原始图片进行可视化
    try:
        img = Image.open(image_path).convert("RGB")
        # 如果图片太大，先缩小再可视化
        w, h = img.size
        if max(w, h) > 2000:
            scale = 2000 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            # 同时缩放框坐标
            for box in boxes:
                for point in box["poly"]:
                    point[0] = int(point[0] * scale)
                    point[1] = int(point[1] * scale)
        
        draw = ImageDraw.Draw(img)
        for b in boxes:
            poly = b["poly"]
            # 绘制四边形
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

# ---------- 转 PAGE-XML（最小可用版） ----------
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

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",     required=True, help="输入图片档路径")
    ap.add_argument("--outdir",    default="out", help="输出文件夹")
    ap.add_argument("--lang",      default="ch",  help="语言代码(ch=中文通用）")
    ap.add_argument("--to_pagexml", action="store_true", help="输出 PAGE-XML")
    ap.add_argument("--with_rec",   action="store_true", help="在 PAGE-XML 中预填文字")
    ap.add_argument("--max_side",   type=int, default=1600, help="图像最大边长（默认1500）")
    ap.add_argument("--preprocess", action="store_true", help="应用图像预处理（增强对比度）")
    args = ap.parse_args()

    try:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

        # drop_rec=True → 只分割；False → 分割+文字
        print("[INFO] 开始侦测文本框...")
        boxes, scale = detect_boxes(args.image, lang=args.lang, drop_rec=(not args.with_rec), preprocess=args.preprocess)
        print(f"[INFO] 侦测到 {len(boxes)} 个文本框")

        if not boxes:
            print("[WARNING] 未检测到任何文本框，程序退出")
            return

        json_path = os.path.join(args.outdir, Path(args.image).stem + ".det.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 侦测结果已输出：{json_path}")

        vis_path = os.path.join(args.outdir, Path(args.image).stem + "_det_vis.jpg")
        visualize_boxes(args.image, boxes, vis_path)
        print(f"[OK] 侦测框可视化已输出：{vis_path}")

        if args.to_pagexml:
            boxes_sorted = sort_vertical_rtl(boxes)
            xml_path = os.path.join(args.outdir, Path(args.image).stem + ".page.xml")
            to_pagexml(args.image, boxes_sorted, xml_path, with_rec=args.with_rec)
            print(f"[OK] PAGE-XML 已输出：{xml_path}")
            print("    汇入 eScriptorium: Images → Import → Transcription (XML)，")
            print("    然后在 Segmentation 面板校正行框/基线，储存后即成为你的 GT。")
            
    except Exception as e:
        print(f"[ERROR] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保取消所有的超时设置
        signal.alarm(0)

if __name__ == "__main__":
    main()
