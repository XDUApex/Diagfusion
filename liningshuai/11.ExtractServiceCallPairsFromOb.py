import os
import json
import pytz
from datetime import datetime

# 定义数据目录和目标保存路径
data_dir = '/home/fuxian/lixuyang/trace_temp_tt'
save_dir = '/home/fuxian/lixuyang/DiagFusion25+tt/liningshuai'
save_path = os.path.join(save_dir, 'services_pairs_tt.json')

# 定义要处理的日期
days = ['2023-01-29', '2023-01-30']

# 存储调用关系的字典
services_pairs = {}

# 遍历每个日期的文件夹
for day in days:
    day_path = os.path.join(data_dir, day)
    if not os.path.exists(day_path):
        print(f"Directory not found: {day_path}")
        continue
    
    print(f"Processing directory: {day_path}")
    files = [f for f in os.listdir(day_path) if f.endswith('.csv')]
    print(f"Found {len(files)} CSV files: {files[:5]}{'...' if len(files) > 5 else ''}")
    
    # 遍历该日期文件夹下的所有 CSV 文件
    for filename in files:
        print(f"Processing file: {filename}")
        try:
            # 文件名格式如 ts-travel-service_ts-contacts-service.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) < 2:
                print(f"Skipping invalid filename: {filename}")
                continue
            
            caller = parts[0]
            callee = parts[1]
            # 去掉可能的实例编号
            caller_base = caller.rsplit('-', 1)[0] if '-' in caller else caller
            callee_base = callee.rsplit('-', 1)[0] if '-' in callee else callee
            
            # 更新 services_pairs 字典
            if caller_base not in services_pairs:
                services_pairs[caller_base] = []
            if callee_base not in services_pairs[caller_base]:
                services_pairs[caller_base].append(callee_base)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 保存到 JSON 文件
with open(save_path, 'w') as f:
    json.dump(services_pairs, f, indent=4)

print(f"Services pairs saved to: {save_path}")
print("Generated services_pairs structure:")
print(json.dumps(services_pairs, indent=4))