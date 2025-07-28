import json
import pickle
import ast

# 读取 JSON 文件
input_json_path = '/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/三模态_tt.json'
output_pkl_path = '/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/三模态_tt.pkl'

with open(input_json_path, 'r') as f:
    json_data = json.load(f)

# 转换数据为 PKL 格式
pkl_data = []
for trace_id, trace_data in json_data.items():
    for key, event_text in trace_data.items():
        try:
            # 解析字符串形式的元组键
            node, anomaly_type = ast.literal_eval(key)
            # 存储为 (trace_id, node_info, text) 格式
            pkl_data.append((trace_id, (node, anomaly_type), event_text))
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing key {key}: {e}")
            continue

# 保存到 PKL 文件
with open(output_pkl_path, 'wb') as f:
    pickle.dump(pkl_data, f)

print(f"PKL file generated at {output_pkl_path}")
print(f"Sample data: {pkl_data[:2]}")