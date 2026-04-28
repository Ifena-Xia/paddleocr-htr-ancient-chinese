try:
    from paddleocr import PaddleOCR
    import inspect
    
    print("=== PaddleOCR API 诊断 ===")
    
    # 检查PaddleOCR类
    ocr = PaddleOCR(lang='ch')
    print(f"✓ PaddleOCR初始化成功")
    
    # 检查ocr方法的参数
    if hasattr(ocr, 'ocr'):
        sig = inspect.signature(ocr.ocr)
        print(f"ocr方法参数: {list(sig.parameters.keys())}")
    else:
        print("✗ 没有找到ocr方法")
        
    # 检查版本
    try:
        import paddleocr
        print(f"PaddleOCR版本: {paddleocr.__version__}")
    except:
        print("无法获取PaddleOCR版本")
        
except Exception as e:
    print(f"诊断失败: {e}")