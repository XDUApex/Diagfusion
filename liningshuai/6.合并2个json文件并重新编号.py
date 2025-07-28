import json
import os
from collections import OrderedDict

def merge_json_files(file_paths, output_path):
    """
    合并多个JSON文件并重新编号
    
    Args:
        file_paths: JSON文件路径列表（按顺序）
        output_path: 输出合并后的JSON文件路径
    
    Returns:
        dict: 合并后的JSON数据
    """
    merged_data = {}
    current_index = 0
    
    print("开始合并JSON文件...")
    
    for file_path in file_paths:
        print(f"\n处理文件: {file_path}")
        
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  成功读取，包含 {len(data)} 个故障记录")
            
            # 统计当前文件的日志数量
            total_logs = sum(len(triplets) for triplets in data.values())
            print(f"  总日志条数: {total_logs}")
            
            # 重新编号并添加到合并数据中
            for old_key in sorted(data.keys(), key=int):  # 按数字顺序排序
                triplets = data[old_key]
                merged_data[str(current_index)] = triplets
                
                print(f"    故障 {old_key} -> 新编号 {current_index} (包含 {len(triplets)} 条日志)")
                current_index += 1
                
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue
    
    # 保存合并后的数据
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"\n合并完成！结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None
    
    return merged_data

def print_merge_summary(merged_data, file_paths):
    """
    打印合并结果摘要
    """
    print("\n=== 合并结果摘要 ===")
    print(f"合并文件数量: {len(file_paths)}")
    print(f"总故障记录数: {len(merged_data)}")
    
    total_logs = sum(len(triplets) for triplets in merged_data.values())
    print(f"总日志条数: {total_logs}")
    
    # 显示每个原文件对应的新编号范围
    current_index = 0
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        
        # 计算这个文件的记录数量（需要重新读取）
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            file_count = len(data)
            
            if file_count > 0:
                start_idx = current_index
                end_idx = current_index + file_count - 1
                print(f"{file_name}: 编号 {start_idx} - {end_idx}")
                current_index += file_count
            else:
                print(f"{file_name}: 无数据")
                
        except Exception as e:
            print(f"{file_name}: 读取失败 - {e}")
    
    print("\n=== 前几条数据示例 ===")
    for key in list(merged_data.keys())[:3]:
        triplets = merged_data[key]
        print(f"故障 {key}: {len(triplets)} 条日志")
        if len(triplets) > 0:
            # 显示第一条日志的时间戳（转换为可读格式）
            from datetime import datetime
            timestamp = triplets[0][0]
            readable_time = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
            service = triplets[0][1]
            print(f"  首条日志: {readable_time}, 服务: {service}")

# 主程序
if __name__ == "__main__":
    # 定义文件路径（按日期顺序）
    base_path = "/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库"
    file_paths = [
        f"{base_path}/2023-01-29-extracted_logs_offset_8h.json",
        f"{base_path}/2023-01-30-extracted_logs_offset_8h.json", 
    ]
    
    # 输出文件路径
    output_path = f"{base_path}/merged_extracted_logs.json"
    
    # 检查文件是否存在
    print("检查文件存在性:")
    for file_path in file_paths:
        exists = os.path.exists(file_path)
        print(f"  {os.path.basename(file_path)}: {'存在' if exists else '不存在'}")
    
    # 执行合并
    merged_data = merge_json_files(file_paths, output_path)
    
    if merged_data is not None:
        # 打印摘要
        print_merge_summary(merged_data, file_paths)
    else:
        print("合并失败！")