import numpy as np
import json
import os
from datetime import datetime

def numpy_encoder(obj):
    """处理NumPy数据类型，将其转换为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def convert_npy_to_json(input_file, output_file=None, indent=2):
    """将.npy文件转换为JSON文件
    
    Args:
        input_file: npy文件路径
        output_file: 输出的json文件路径，默认为与输入文件同名但扩展名为.json
        indent: JSON格式的缩进空格数
    """
    # 如果未指定输出文件，创建默认名称
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.json'
    
    try:
        # 加载NumPy数组
        data = np.load(input_file, allow_pickle=True)
        
        # 打印数组基本信息，帮助理解数据
        print(f"数组形状: {data.shape}")
        print(f"数组类型: {data.dtype}")
        print(f"数组元素数量: {len(data)}")
        
        # 处理多元素对象数组 - 修复部分
        json_data = {}
        for i, item in enumerate(data):
            json_data[str(i)] = item
        
        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=indent, default=numpy_encoder)
        
        print(f"转换成功! 已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 在此直接指定输入和输出文件路径
    input_file = '/home/fuxian/lixuyang/fuxian/DiagFusion25+AIOps/data/gaia/demo/demo_1100/anomalies/stratification_logs.npy'  # 替换为你的.npy文件路径
    output_file = '/home/fuxian/lixuyang/fuxian/DiagFusion25+AIOps/data/gaia/demo/demo_1100/anomalies/stratification_logs.json'          # 替换为你想要的输出路径
    indent = 2                                         # JSON缩进空格数
    
    # 执行转换
    convert_npy_to_json(input_file, output_file, indent)
