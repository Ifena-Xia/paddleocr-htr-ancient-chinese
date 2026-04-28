import os
import xml.etree.ElementTree as ET

def merge_xml_files(folder_path, output_file="merged.xml"):
    """
    合并文件夹中的所有XML文件
    
    Args:
        folder_path: 包含XML文件的文件夹路径
        output_file: 输出文件路径
    """
    # 创建新的根元素
    merged_root = ET.Element("merged_data")
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            print(f"处理文件: {filename}")
            
            try:
                # 解析XML文件
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # 创建当前文件的包装元素（可选）
                file_element = ET.Element("source_file")
                file_element.set("name", filename)
                
                # 复制原始根元素的所有内容到包装元素
                for child in root:
                    file_element.append(child)
                
                # 添加到合并的根元素
                merged_root.append(file_element)
                
            except ET.ParseError as e:
                print(f"  警告: 无法解析文件 {filename}, 错误: {e}")
                continue
    
    # 创建ElementTree对象并写入文件
    merged_tree = ET.ElementTree(merged_root)
    
    # 美化输出（添加缩进）
    indent(merged_root)
    
    # 写入文件
    merged_tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"\n合并完成！结果已保存到: {output_file}")
    print(f"共合并了 {len(merged_root)} 个文件")

def indent(elem, level=0):
    """
    为XML元素添加缩进，美化输出
    """
    indent_str = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str

# 使用示例
if __name__ == "__main__":
    # 设置你的文件夹路径（桌面上的文件夹）
    # 例如：folder_path = r"C:\Users\你的用户名\Desktop\你的文件夹"
    folder_path = input("请输入包含XML文件的文件夹路径: ").strip()
    
    # 验证路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 路径 '{folder_path}' 不存在")
    else:
        # 调用合并函数
        output_file = os.path.join(folder_path, "merged.xml")
        merge_xml_files(folder_path, output_file)