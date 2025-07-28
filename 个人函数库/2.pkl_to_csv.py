import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_pkl(file_path):
    """加载pkl文件"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_as(data, output_path, format_type='json'):
    """将数据保存为指定格式"""
    output_path = Path(output_path)
    
    if format_type == 'json':
        # 转换元组键为字符串
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_keys(x) for x in obj]
            return obj
            
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(convert_keys(data), f, indent=4)
    elif format_type == 'csv':
        if isinstance(data, dict):
            pd.DataFrame.from_dict(data).to_csv(output_path.with_suffix('.csv'))
        else:
            pd.DataFrame(data).to_csv(output_path.with_suffix('.csv'))
    elif format_type == 'txt':
        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write(str(data))
    elif format_type == 'npy':
        np.save(output_path.with_suffix('.npy'), data)
    else:
        raise ValueError(f"不支持的格式: {format_type}")

# 使用示例
if __name__ == '__main__':
    # 示例路径 - 替换为您的实际pkl文件路径
    pkl_file = 'D:\科研PART\本科毕设\DiagFusion\DiagFusion-378D\data\gaia\demo\demo_1100\parse\stratification_texts.pkl'
    
    # 加载pkl文件
    data = load_pkl(pkl_file)
    print(f"成功加载pkl文件，数据类型: {type(data)}")
    
    # 保存为JSON
    save_as(data, pkl_file, 'json')
    print("已转换为JSON格式")
    
    # 保存为CSV（如果是结构化数据）
    if isinstance(data, (dict, list, np.ndarray)):
        save_as(data, pkl_file, 'csv')
        print("已转换为CSV格式")