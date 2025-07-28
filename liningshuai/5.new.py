import pandas as pd
import json
from datetime import datetime, timedelta
import pytz
import os
import re

def datetime_to_timestamp(datetime_str, add_8_hours=False):
    """
    将日期时间字符串（假设为 UTC+8）转换为 UTC 时间戳，可选择是否加8小时偏移
    
    Args:
        datetime_str: 格式如 "2023-01-29 08:43:04.000000" 或 ISO格式（如 "2023-01-29T10:05:28.579588645Z"）
        add_8_hours: 是否在时间上加8小时
    
    Returns:
        float: UTC 时间戳
    """
    try:
        tz = pytz.timezone('Asia/Shanghai')
        
        if ' ' in datetime_str and '.' in datetime_str:
            dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        elif 'T' in datetime_str and 'Z' in datetime_str:
            if '.' in datetime_str:
                dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")
        else:
            raise ValueError(f"Unsupported datetime format: {datetime_str}")
        
        if add_8_hours:
            dt = dt + timedelta(hours=8)
        
        dt = tz.localize(dt).astimezone(pytz.UTC)
        return dt.timestamp()
    except Exception as e:
        print(f"时间转换失败: {datetime_str}, 错误: {e}")
        return None

def extract_logs_by_groundtruth(groundtruth_path, log_dir, output_path_original, output_path_offset):
    """
    根据groundtruth.csv中的时间范围，从log目录中的多个日志文件提取数据
    生成两个输出：原始时间和加8小时偏移时间
    
    Args:
        groundtruth_path: groundtruth.csv的文件路径
        log_dir: log文件的目录路径
        output_path_original: 原始时间输出的JSON文件路径
        output_path_offset: 加8小时偏移输出的JSON文件路径
    
    Returns:
        tuple: (原始时间的JSON数据, 加8小时偏移的JSON数据)
    """
    
    # 读取groundtruth文件
    try:
        groundtruth_df = pd.read_csv(groundtruth_path)
        print(f"成功读取groundtruth文件，共 {len(groundtruth_df)} 条记录")
        print(f"Groundtruth列名: {list(groundtruth_df.columns)}")
        print("groundtruth st_time 示例:", groundtruth_df['st_time'].head().tolist())
        print("groundtruth ed_time 示例:", groundtruth_df['ed_time'].head().tolist())
    except Exception as e:
        print(f"读取groundtruth文件失败: {e}")
        return None, None
    
    # 遍历log目录下的所有csv文件
    all_logs_df = pd.DataFrame()
    for filename in os.listdir(log_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(log_dir, filename)
            try:
                df = pd.read_csv(file_path, header=0)
                all_logs_df = pd.concat([all_logs_df, df], ignore_index=True)
            except Exception as e:
                print(f"读取 {filename} 失败: {e}")
                continue
    
    if all_logs_df.empty:
        print("未成功读取任何日志文件")
        return None, None
    
    print(f"成功读取 {len(all_logs_df)} 条日志记录")
    print(f"Log文件列名: {list(all_logs_df.columns)}")
    
    # 使用 TimeUnixNano 转换为毫秒时间戳
    all_logs_df['timestamp_ms'] = all_logs_df['TimeUnixNano'] / 1e6
    
    print(f"log_df timestamp_ms 范围: {all_logs_df['timestamp_ms'].min()} - {all_logs_df['timestamp_ms'].max()}")
    
    # 处理两种情况：原始时间和加8小时偏移
    result_json_original = {}
    result_json_offset = {}
    
    for case, add_8_hours, result_json, label in [
        (0, False, result_json_original, "原始时间"),
        (1, True, result_json_offset, "加8小时偏移")
    ]:
        # 将groundtruth中的时间转换为时间戳
        groundtruth_df[f'st_timestamp_{case}'] = groundtruth_df['st_time'].apply(
            lambda x: datetime_to_timestamp(x, add_8_hours=add_8_hours))
        groundtruth_df[f'ed_timestamp_{case}'] = groundtruth_df['ed_time'].apply(
            lambda x: datetime_to_timestamp(x, add_8_hours=add_8_hours))
        
        # 检查转换结果
        print(f"\n{label} 时间转换示例:")
        for i in range(min(3, len(groundtruth_df))):
            row = groundtruth_df.iloc[i]
            print(f"故障 {i}:")
            print(f"  原始开始时间: {row['st_time']}")
            print(f"  转换后时间戳: {row[f'st_timestamp_{case}']}")
            print(f"  原始结束时间: {row['ed_time']}")
            print(f"  转换后时间戳: {row[f'ed_timestamp_{case}']}")
            print()
        
        # 遍历每个故障时间段
        for idx, row in groundtruth_df.iterrows():
            st_timestamp = row[f'st_timestamp_{case}']
            ed_timestamp = row[f'ed_timestamp_{case}']
            
            if st_timestamp is None or ed_timestamp is None:
                print(f"{label} 跳过第 {idx} 个时间段（时间转换失败）")
                continue
            
            print(f"{label} 处理第 {idx} 个时间段:")
            print(f"  时间范围: {row['st_time']} - {row['ed_time']}")
            print(f"  时间戳范围 (UTC): {st_timestamp} - {ed_timestamp}")
            
            # 转换为毫秒时间戳以匹配 log 数据
            st_timestamp_ms = int(st_timestamp * 1000)
            ed_timestamp_ms = int(ed_timestamp * 1000)
            
            # 过滤日志
            filtered_logs = all_logs_df[
                (all_logs_df['timestamp_ms'] >= st_timestamp_ms) & 
                (all_logs_df['timestamp_ms'] <= ed_timestamp_ms)
            ]
            
            print(f"  找到 {len(filtered_logs)} 条日志记录")
            
            # 构建三元组列表
            triplets = []
            for _, log_row in filtered_logs.iterrows():
                try:
                    # 提取有效的 JSON 部分
                    log_str = log_row['Log'].strip()
                    match = re.match(r'^{.*}', log_str, re.DOTALL)
                    if match:
                        log_str = match.group(0)
                    log_json = json.loads(log_str)
                    log_content = log_row['Log']  # 使用原始 Log 列内容
                    
                    timestamp_ms = int(log_row['timestamp_ms'])
                    pod_name = log_row['PodName']
                    
                    triplet = [
                        timestamp_ms,
                        pod_name,
                        log_content
                    ]
                    triplets.append(triplet)
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"解析日志失败: {log_row['Log']}, 错误: {e}")
                    continue
            
            # 按时间戳排序
            triplets.sort(key=lambda x: x[0])
            
            # 添加到结果字典
            result_json[str(idx)] = triplets
            print(f"  故障 {idx} 处理完成\n")
    
    # 保存原始时间结果
    try:
        os.makedirs(os.path.dirname(output_path_original), exist_ok=True)
        with open(output_path_original, 'w', encoding='utf-8') as f:
            json.dump(result_json_original, f, ensure_ascii=False, indent=2)
        print(f"原始时间结果已保存到: {output_path_original}")
    except Exception as e:
        print(f"保存原始时间文件失败: {e}")
    
    # 保存加8小时偏移结果
    try:
        os.makedirs(os.path.dirname(output_path_offset), exist_ok=True)
        with open(output_path_offset, 'w', encoding='utf-8') as f:
            json.dump(result_json_offset, f, ensure_ascii=False, indent=2)
        print(f"加8小时偏移结果已保存到: {output_path_offset}")
    except Exception as e:
        print(f"保存加8小时偏移文件失败: {e}")
    
    return result_json_original, result_json_offset

def print_summary(result_json, label):
    """
    打印结果摘要信息
    
    Args:
        result_json: JSON数据
        label: 摘要标签（原始时间或加8小时偏移）
    """
    print(f"\n=== {label} 提取结果摘要 ===")
    total_logs = 0
    for key, triplets in result_json.items():
        count = len(triplets)
        total_logs += count
        print(f"故障 {key}: {count} 条日志")
        
        if count > 0:
            min_time = min(triplet[0] for triplet in triplets)
            max_time = max(triplet[0] for triplet in triplets)
            pods = set(triplet[1] for triplet in triplets)
            print(f"  时间范围: {min_time} - {max_time}")
            print(f"  涉及 PodName: {', '.join(list(pods)[:5])}{'...' if len(pods) > 5 else ''}")
        print()
    
    print(f"{label} 总计提取日志: {total_logs} 条")

# 主程序
if __name__ == "__main__":
    groundtruth_path = "/home/fuxian/DataSet/tt/2023-01-30/groundtruth.csv"
    log_dir = "/home/fuxian/DataSet/tt/2023-01-30/log"
    output_path_original = "/home/fuxian/lixuyang/DiagFusion25+tt/liningshuai/2023-01-30-extracted_logs_original.json"
    output_path_offset = "/home/fuxian/lixuyang/DiagFusion25+tt/liningshuai/2023-01-30-extracted_logs_offset_8h.json"
    print("开始提取日志数据...")
    
    result_original, result_offset = extract_logs_by_groundtruth(
        groundtruth_path, log_dir, output_path_original, output_path_offset
    )
    
    if result_original is not None and result_offset is not None:
        print_summary(result_original, "原始时间")
        print_summary(result_offset, "加8小时偏移")
        
        for label, result in [("原始时间", result_original), ("加8小时偏移", result_offset)]:
            print(f"\n=== {label} 数据示例 ===")
            for key, triplets in list(result.items())[:2]:
                print(f"\n故障 {key} 的前3条数据:")
                for i, triplet in enumerate(triplets[:3]):
                    timestamp_readable = datetime.fromtimestamp(
                        triplet[0]/1000, tz=pytz.timezone('Asia/Shanghai')
                    ).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    print(f"  [{triplet[0]}, \"{triplet[1]}\", \"{triplet[2][:50]}...\"]")
                    print(f"    时间: {timestamp_readable}")
                if len(triplets) > 3:
                    print(f"  ... 还有 {len(triplets) - 3} 条数据")
    else:
        print("提取失败！")