import os
import json
import argparse
from pathlib import Path

def extract_csv_filenames(folder_path, include_subfolders=True, print_full_path=False):
    """
    提取指定文件夹下的所有CSV文件的名称
    
    Args:
        folder_path: 要搜索的文件夹路径
        include_subfolders: 是否包含子文件夹中的CSV文件
        print_full_path: 是否打印完整路径而非仅文件名
        
    Returns:
        csv_files: 包含所有CSV文件名的列表
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 指定的文件夹 '{folder_path}' 不存在")
        return []
    
    csv_files = []
    
    # 根据是否包含子文件夹选择搜索方法
    if include_subfolders:
        # 递归遍历所有子文件夹
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.csv'):
                    if print_full_path:
                        csv_files.append(os.path.join(root, file))
                    else:
                        csv_files.append(file)
    else:
        # 只遍历当前文件夹
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith('.csv'):
                if print_full_path:
                    csv_files.append(file_path)
                else:
                    csv_files.append(file)
    
    return csv_files

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='提取指定文件夹下的所有CSV文件名')
    parser.add_argument('folder_path', help='要搜索的文件夹路径')
    parser.add_argument('--no-subfolders', action='store_true', help='不包含子文件夹中的CSV文件')
    parser.add_argument('--full-path', action='store_true', help='显示完整文件路径')
    parser.add_argument('--output', default='csv_files.json', help='JSON输出文件路径 (默认: csv_files.json)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 提取CSV文件名
    csv_files = extract_csv_filenames(
        args.folder_path, 
        not args.no_subfolders,
        args.full_path
    )
    
    # 将结果保存为JSON文件
    output_file = args.output
    
    # 创建包含更多信息的JSON对象
    json_data = {
        "folder_path": args.folder_path,
        "include_subfolders": not args.no_subfolders,
        "full_path": args.full_path,
        "csv_count": len(csv_files),
        "csv_files": csv_files
    }
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"已将 {len(csv_files)} 个CSV文件名保存到: {output_file}")

if __name__ == "__main__":
    main()