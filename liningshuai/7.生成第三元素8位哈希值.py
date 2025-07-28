import json
import hashlib
import pandas as pd
import os
import sys
import tempfile

# 设置项目根目录路径
PROJECT_ROOT = r"/home/fuxian/lixuyang/fuxian/DiagFusion25+tt"
sys.path.insert(0, PROJECT_ROOT)

# 根据检测结果，正确的导入路径
try:
    from log.logparser.logparser.Drain.Drain import LogParser
    print("成功导入Drain模块")
except ImportError as e:
    print(f"导入失败: {e}")
    # 尝试另一种导入方式
    try:
        # 直接添加Drain目录到路径
        drain_path = os.path.join(PROJECT_ROOT, 'log', 'logparser', 'logparser', 'Drain')
        sys.path.insert(0, drain_path)
        from Drain import LogParser
        print("成功导入Drain模块 - 备用方式")
    except ImportError as e2:
        print(f"备用导入也失败: {e2}")
        print("将使用简化的哈希生成方案")
        LogParser = None

def create_temp_csv_from_json(data):
    """
    将JSON数据转换为临时CSV文件供Drain处理
    """
    # 创建临时文件
    temp_fd, temp_csv_path = tempfile.mkstemp(suffix='.csv', prefix='temp_logs_')
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            # 写入CSV头部
            temp_file.write("LineId,timestamp,cmdb_id,log_name,Content\n")
            
            line_id = 1
            for case_id, logs in data.items():
                if isinstance(logs, list):
                    for log_entry in logs:
                        if len(log_entry) >= 3:
                            timestamp = log_entry[0]
                            service_name = log_entry[1]
                            content = str(log_entry[2]).replace(',', ' ').replace('\n', ' ')
                            
                            # 写入CSV行
                            temp_file.write(f"{line_id},{timestamp},{case_id},{service_name},{content}\n")
                            line_id += 1
    except Exception as e:
        # 如果出错，删除临时文件
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        raise e
    
    return temp_csv_path

def generate_hash_mapping(templates_file, structured_file, original_data):
    """
    从Drain输出文件生成哈希值映射
    """
    try:
        # 读取模板文件
        templates_df = pd.read_csv(templates_file)
        structured_df = pd.read_csv(structured_file)
        
        # 创建EventId到哈希值的映射
        template_to_hash = {}
        for _, row in templates_df.iterrows():
            event_template = row['EventTemplate']
            # 生成8位哈希值
            hash_8bit = hashlib.md5(event_template.encode('utf-8')).hexdigest()[:8]
            template_to_hash[row['EventId']] = {
                'hash_8bit': hash_8bit,
                'template': event_template
            }
        
        # 构建结果
        result = {}
        for case_id, logs in original_data.items():
            result[case_id] = []
            if isinstance(logs, list):
                log_index = 0
                for log_entry in logs:
                    if len(log_entry) >= 3:
                        timestamp, service_name, content = log_entry[0], log_entry[1], log_entry[2]
                        
                        # 查找对应的EventId
                        event_id = 'unknown'
                        hash_8bit = 'unknown'
                        template = str(content)
                        
                        # 在structured_df中查找匹配的行
                        matching_rows = structured_df[
                            (structured_df['cmdb_id'] == case_id) & 
                            (structured_df['LineId'] == log_index + 1)
                        ]
                        
                        if not matching_rows.empty:
                            event_id = matching_rows.iloc[0]['EventId']
                            if event_id in template_to_hash:
                                hash_8bit = template_to_hash[event_id]['hash_8bit']
                                template = template_to_hash[event_id]['template']
                        
                        result[case_id].append({
                            'timestamp': timestamp,
                            'service_name': service_name,
                            'original_content': content,
                            'event_id': event_id,
                            'hash_8bit': hash_8bit,
                            'template': template
                        })
                        
                        log_index += 1
        
        return result
        
    except Exception as e:
        print(f"生成哈希值映射时出错: {e}")
        return None

def print_sample_results(result):
    """
    打印样本结果
    """
    print("\n=== 样本结果 ===")
    sample_count = 0
    for case_id, logs in result.items():
        if sample_count >= 3:  # 只显示前3个案例
            break
        print(f"\n案例 {case_id}:")
        for i, log in enumerate(logs[:2]):  # 每个案例只显示前2条日志
            print(f"  日志 {i+1}:")
            print(f"    时间戳: {log['timestamp']}")
            print(f"    服务名: {log['service_name']}")
            print(f"    哈希值: {log['hash_8bit']}")
            print(f"    原始内容: {str(log['original_content'])[:100]}...")
        if len(logs) > 2:
            print(f"    ... 还有 {len(logs) - 2} 条日志")
        sample_count += 1

def save_result_to_json(result, output_file):
    """
    保存结果到JSON文件
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

def process_json_logs_with_drain(json_file_path, output_dir='./drain_result/'):
    """
    处理JSON格式的日志文件，使用Drain算法生成8位哈希值
    
    Args:
        json_file_path: JSON文件路径
        output_dir: 输出目录
    
    Returns:
        dict: 处理后的结果，包含8位哈希值
    """
    print(f"开始处理文件: {json_file_path}")
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON文件，包含 {len(data)} 个案例")
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return None
    
    # 如果Drain模块导入失败，使用简化方案
    if LogParser is None:
        print("使用简化的哈希生成方案...")
        return process_logs_simple_hash(data)
    
    # 创建临时CSV文件供Drain处理
    temp_csv_path = create_temp_csv_from_json(data)
    print(f"创建临时CSV文件: {temp_csv_path}")
    
    # 配置Drain参数
    log_format = '<LineId>,<timestamp>,<cmdb_id>,<log_name>,<Content>'
    regex = [
        r'blk_(|-)[0-9]+',  # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        r'[A-Fa-f0-9]{8,}',  # 十六进制数
        r'\d{4}-\d{2}-\d{2}',  # 日期格式
        r'\d{2}:\d{2}:\d{2}',  # 时间格式
    ]
    st = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    try:
        # 初始化Drain解析器
        parser = LogParser(
            log_format=log_format,
            indir=os.path.dirname(temp_csv_path),
            outdir=output_dir,
            depth=depth, 
            st=st, 
            rex=regex
        )
        
        # 解析日志
        csv_filename = os.path.basename(temp_csv_path)
        print(f"开始Drain解析，文件: {csv_filename}")
        parser.parse(csv_filename)
        print("Drain解析完成")
        
        # 读取解析结果并生成8位哈希值
        templates_file = os.path.join(output_dir, f'{csv_filename}_templates.csv')
        structured_file = os.path.join(output_dir, f'{csv_filename}_structured.csv')
        
        if os.path.exists(templates_file) and os.path.exists(structured_file):
            result = generate_hash_mapping(templates_file, structured_file, data)
            print("成功生成哈希值映射")
        else:
            print("Drain输出文件不存在，检查解析过程")
            return None
            
    except Exception as e:
        print(f"Drain解析过程出错: {e}")
        print("切换到简化方案...")
        return process_logs_simple_hash(data)
    finally:
        # 清理临时文件
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            print(f"清理临时文件: {temp_csv_path}")
    
    return result

def process_logs_simple_hash(data):
    """
    简化的哈希生成方案，不依赖Drain
    """
    import re
    
    print("使用简化哈希生成方案...")
    result = {}
    
    # 简单的模板提取正则表达式
    template_regex = [
        (r'\d{4}-\d{2}-\d{2}', '<DATE>'),
        (r'\d{2}:\d{2}:\d{2}', '<TIME>'),
        (r'\d+\.\d+\.\d+\.\d+', '<IP>'),
        (r'\d+', '<NUM>'),
        (r'[a-fA-F0-9]{8,}', '<HEX>'),
    ]
    
    for case_id, logs in data.items():
        result[case_id] = []
        if isinstance(logs, list):
            for log_entry in logs:
                if len(log_entry) >= 3:
                    timestamp, service_name, content = log_entry[0], log_entry[1], log_entry[2]
                    
                    # 简单的模板化
                    content_str = str(content)
                    template = content_str
                    
                    # 应用正则替换生成模板
                    for pattern, replacement in template_regex:
                        template = re.sub(pattern, replacement, template)
                    
                    # 生成基于模板的8位哈希值
                    hash_8bit = hashlib.md5(template.encode('utf-8')).hexdigest()[:8]
                    
                    result[case_id].append({
                        'timestamp': timestamp,
                        'service_name': service_name,
                        'original_content': content,
                        'event_id': f'simple_{hash_8bit}',
                        'hash_8bit': hash_8bit,
                        'template': template
                    })
    
    return result

# 主程序入口
if __name__ == "__main__":
    # 配置文件路径 - 请修改为您的实际文件路径
    json_input_file = "merged_extracted_logs.json"  # 输入的JSON文件
    json_output_file = "哈希值__extracted_logs.json"  # 输出的JSON文件
    
    # 检查当前目录下是否有JSON文件
    current_dir_files = [f for f in os.listdir('.') if f.endswith('.json')]
    if current_dir_files and json_input_file == "your_log_file.json":
        print("发现当前目录下的JSON文件:")
        for i, file in enumerate(current_dir_files):
            print(f"  {i+1}. {file}")
        
        # 如果只有一个JSON文件，自动使用它
        if len(current_dir_files) == 1:
            json_input_file = current_dir_files[0]
            print(f"自动选择文件: {json_input_file}")
    
    print("=== 日志哈希值生成工具 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输入文件: {json_input_file}")
    print(f"输出文件: {json_output_file}")
    print()
    
    # 检查输入文件是否存在
    if not os.path.exists(json_input_file):
        print(f"错误: 输入文件不存在 - {json_input_file}")
        print("请修改 json_input_file 变量为正确的文件路径")
        print(f"当前目录: {os.getcwd()}")
        print(f"当前目录下的文件: {os.listdir('.')}")
        sys.exit(1)
    
    # 处理日志文件
    result = process_json_logs_with_drain(json_input_file)
    
    if result:
        # 打印样本结果
        print_sample_results(result)
        
        # 保存完整结果
        save_result_to_json(result, json_output_file)
        
        print(f"\n=== 处理完成 ===")
        print(f"成功处理并保存结果到: {json_output_file}")
        
        # 统计信息
        total_logs = sum(len(logs) for logs in result.values())
        unique_hashes = set()
        for logs in result.values():
            for log in logs:
                if log['hash_8bit'] != 'unknown':
                    unique_hashes.add(log['hash_8bit'])
        
        print(f"总计处理日志: {total_logs} 条")
        print(f"生成唯一哈希: {len(unique_hashes)} 个")
        
    else:
        print("处理失败，请检查错误信息")