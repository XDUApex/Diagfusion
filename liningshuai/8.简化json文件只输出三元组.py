import json

# 输入和输出文件路径
input_file = "哈希值__extracted_logs.json"  # 替换为您的输入 JSON 文件路径
output_file = "8.简化三元组_extracted_logs.json"  # 替换为您想要的输出 JSON 文件路径

# 读取输入 JSON 文件
with open(input_file, 'r') as f:
    data = json.load(f)

# 处理数据，提取所需字段
output_data = {}
for key, entries in data.items():
    output_data[key] = [
        [entry["timestamp"], entry["service_name"], entry["hash_8bit"]]
        for entry in entries
    ]

# 保存到新的 JSON 文件
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"处理完成，结果已保存到 {output_file}")