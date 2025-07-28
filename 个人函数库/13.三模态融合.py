# -*- coding: utf-8 -*-
import json
import math
import public_function as pf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def load_groundtruth(groundtruth_paths):
    """
    加载所有日期的 groundtruth.csv 文件，合并为一个 DataFrame
    groundtruth_paths: 包含各日期 groundtruth.csv 文件路径的列表
    """
    all_labels = []
    for path in groundtruth_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_labels.append(df)
            print(f"Loaded {len(df)} cases from {path}")
        else:
            print(f"Warning: File not found: {path}")
    
    if all_labels:
        labels = pd.concat(all_labels, ignore_index=True)
        print(f"Total loaded: {len(labels)} fault cases")
        return labels
    else:
        print("No groundtruth files found!")
        return pd.DataFrame()

def metric_trace_log_parse(trace, metric, logs, labels, save_path, nodes):
    # 移除 metric 中可能存在的 np.inf 值
    if metric is not None:
        for k, v in metric.items():
            # 过滤掉无效数据和inf值
            metric[k] = [x for x in v if len(x) >= 4 and not math.isinf(x[3])]

    # 确保 logs 格式与 labels 索引对齐
    log = None
    if logs is not None:
        log = {x: logs.get(str(x), []) for x in labels.index}

    # 从 groundtruth.csv 获取实例名和异常类型
    instance_names = sorted(list(set(labels['instance'].apply(lambda x: x.split('-')[0]))))  # 提取服务名
    anomaly_instances = list(labels['instance'])  # 异常实例
    anomaly_type = list(labels['anomaly_type'])  # 异常类型

    print(f"DEBUG: Instance names from labels: {instance_names}")
    print(f"DEBUG: Total cases: {len(labels)}")
    print(f"DEBUG: Sample anomaly instances: {anomaly_instances[:5] if anomaly_instances else 'None'}")
    print(f"DEBUG: Sample anomaly types: {anomaly_type[:5] if anomaly_type else 'None'}")
    
    # 检查数据文件状态
    if metric:
        print(f"DEBUG: Metric data loaded for {len(metric)} cases")
        first_case = list(metric.keys())[0] if metric else None
        if first_case and metric[first_case]:
            print(f"DEBUG: Sample metric data for case {first_case}: {metric[first_case][:2]}")
    
    if trace:
        print(f"DEBUG: Trace data loaded for {len(trace)} cases")
        first_case = list(trace.keys())[0] if trace else None
        if first_case and trace[first_case]:
            print(f"DEBUG: Sample trace data for case {first_case}: {trace[first_case][:2]}")
    
    if log:
        print(f"DEBUG: Log data loaded for {len(log)} cases")
        first_case = list(log.keys())[0] if log else None
        if first_case and log[first_case]:
            print(f"DEBUG: Sample log data for case {first_case}: {log[first_case][:2]}")

    # 检查 nodes 与 instance_names 的一致性
    nodes_list = nodes.split()
    missing_nodes = [n for n in instance_names if n not in nodes_list]
    extra_nodes = [n for n in nodes_list if n not in instance_names]
    print(f"DEBUG: Nodes in config: {nodes_list}")
    print(f"DEBUG: Missing nodes in config (not in labels): {missing_nodes}")
    print(f"DEBUG: Extra nodes in config (not in labels): {extra_nodes}")

    # 初始化输出字典，按 labels 索引构建
    demo_metric = {x: {} for x in labels.index}
    
    # 遍历每个案例
    k = 0
    for case_id, v in tqdm(demo_metric.items(), desc="Processing cases"):
        anomaly_instance_name = anomaly_instances[k]  # 当前案例的异常实例
        anomaly_instance_base = anomaly_instance_name.split('-')[0]  # 提取服务名
        anomaly_instance_type = anomaly_type[k]  # 当前案例的异常类型
        k += 1
        
        # 为每个实例生成键，异常实例标注为具体异常类型，其他为 [normal]
        inner_dict_key = [(x, anomaly_instance_type) if x == anomaly_instance_base else (x, "[normal]") for x in instance_names]
        
        # 初始化每个实例的列表
        for inner_key in inner_dict_key:
            demo_metric[case_id][inner_key] = []
        
        # 处理 metric 模态 - 使用服务名模糊匹配
        if metric is not None and str(case_id) in metric:
            for inner_key in inner_dict_key:
                instance_base = inner_key[0]  # 服务名，如 ts-contacts-service
                matching_metrics = []
                for y in metric[str(case_id)]:
                    if len(y) >= 4:
                        # 模糊匹配：只比较服务名
                        cmdb_id_base = y[1].split('-')[0] if isinstance(y[1], str) else y[1]
                        if instance_base == cmdb_id_base:
                            event_name = "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")
                            matching_metrics.append([y[0], event_name])
                demo_metric[case_id][inner_key].extend(matching_metrics)
                
                # 调试前几个case
                if case_id < 3 and matching_metrics:
                    print(f"DEBUG: Case {case_id}, Instance {instance_base}: Found {len(matching_metrics)} metrics")
        
        # 处理 trace 模态 - 使用服务名模糊匹配
        if trace is not None and str(case_id) in trace:
            for inner_key in inner_dict_key:
                instance_base = inner_key[0]
                matching_traces = []
                for y in trace[str(case_id)]:
                    if len(y) >= 3:
                        # 模糊匹配：检查源服务或目标服务
                        src_base = y[1].split('-')[0] if isinstance(y[1], str) else y[1]
                        dst_base = y[2].split('-')[0] if isinstance(y[2], str) else y[2]
                        if instance_base in (src_base, dst_base):
                            trace_name = "{}_{}".format(y[1], y[2])
                            matching_traces.append([y[0], trace_name])
                demo_metric[case_id][inner_key].extend(matching_traces)
                
                # 调试前几个case
                if case_id < 3 and matching_traces:
                    print(f"DEBUG: Case {case_id}, Instance {instance_base}: Found {len(matching_traces)} traces")
        
        # 处理 log 模态 - 使用服务名模糊匹配
        if log is not None and case_id in log:
            for inner_key in inner_dict_key:
                instance_base = inner_key[0]
                matching_logs = []
                for y in log[case_id]:
                    if len(y) >= 3:
                        # 模糊匹配：检查服务实例名
                        service_base = y[1].split('-')[0] if isinstance(y[1], str) else y[1]
                        if instance_base == service_base:
                            matching_logs.append([y[0], y[2]])
                demo_metric[case_id][inner_key].extend(matching_logs)
                
                # 调试前几个case
                if case_id < 3 and matching_logs:
                    print(f"DEBUG: Case {case_id}, Instance {instance_base}: Found {len(matching_logs)} logs")
        
        # 对每个实例的条目按时间戳排序并连接为字符串
        for inner_key in inner_dict_key:
            temp = demo_metric[case_id][inner_key]
            if temp:  # 只处理非空列表
                sort_list = sorted(temp, key=lambda x: x[0])  # 按时间戳排序
                temp_list = [x[1] for x in sort_list]  # 提取内容
                demo_metric[case_id][inner_key] = ' '.join(temp_list)  # 连接为字符串
            else:
                demo_metric[case_id][inner_key] = ""  # 空字符串

    # 转换 tuple 键为字符串以兼容 JSON
    demo_metric_str_keys = {}
    for case_id, inner_dict in demo_metric.items():
        demo_metric_str_keys[case_id] = {}
        for key_tuple, value in inner_dict.items():
            key_str = str(key_tuple)
            demo_metric_str_keys[case_id][key_str] = value

    # 统计结果
    total_events = 0
    non_empty_cases = 0
    for case_id, inner_dict in demo_metric_str_keys.items():
        case_events = sum(1 for v in inner_dict.values() if v.strip())
        total_events += case_events
        if case_events > 0:
            non_empty_cases += 1
    
    print(f"\n=== Fusion Results Summary ===")
    print(f"Total cases processed: {len(demo_metric_str_keys)}")
    print(f"Cases with events: {non_empty_cases}")
    print(f"Total non-empty event sequences: {total_events}")

    # 保存结果，带缩进以提高可读性
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(demo_metric_str_keys, f, indent=4, ensure_ascii=False)
    print(f"Results saved to: {save_path}")

def run_parse(config, labels):
    trace = None
    metric = None
    logs = None
    
    # 加载三个模态的 JSON 文件
    if config['trace_path']:
        try:
            with open(config['trace_path'], 'r', encoding='utf8') as fp:
                trace = json.load(fp)
            print(f"Trace data loaded from: {config['trace_path']}")
        except Exception as e:
            print(f"Error loading trace data: {e}")
    
    if config['metric_path']:
        try:
            with open(config['metric_path'], 'r', encoding='utf8') as fp:
                metric = json.load(fp)
            print(f"Metric data loaded from: {config['metric_path']}")
        except Exception as e:
            print(f"Error loading metric data: {e}")
    
    if config['log_path']:
        try:
            with open(config['log_path'], 'r', encoding='utf8') as fp:
                logs = json.load(fp)
            print(f"Log data loaded from: {config['log_path']}")
        except Exception as e:
            print(f"Error loading log data: {e}")
    
    # 调用融合函数
    metric_trace_log_parse(trace, metric, logs, labels, config['save_path'], config['nodes'])

if __name__ == '__main__':
    # 配置路径
    config = {
        'trace_path': '/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/trace_events_tt_original.json',
        'metric_path': '/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/metric_events_tt_default_offset_8h.json',
        'log_path': '/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/8.简化三元组_extracted_logs.json',
        'save_path': '/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/三模态_tt.json',
        'nodes': 'ts-auth-service ts-basic-service ts-cancel-service ts-contacts-service ts-delivery-service ts-execute-service ts-food-service ts-gateway-service ts-inside-payment-service ts-order-other-service ts-order-service ts-payment-service ts-preserve-other-service ts-preserve-service ts-price-service ts-route-service ts-seat-service ts-security-service ts-station-food-service ts-station-service ts-train-food-service ts-train-service ts-travel-service ts-travel2-service ts-user-service ts-verification-code-service'
    }
    
    # 加载所有日期的 groundtruth.csv
    dates = ['2023-01-29', '2023-01-30']
    groundtruth_paths = [os.path.join('/home/fuxian/DataSet/NewDataset/tt', date, 'groundtruth.csv') for date in dates]
    labels = load_groundtruth(groundtruth_paths)
    
    if not labels.empty:
        # 显示labels的基本信息
        print(f"\n=== Labels Analysis ===")
        print(f"Columns: {labels.columns.tolist()}")
        print(f"Unique instances: {sorted(labels['instance'].unique().tolist())}")
        print(f"Unique anomaly types: {sorted(labels['anomaly_type'].unique().tolist())}")
        print(f"Sample data:\n{labels.head()}")
        
        # 运行融合
        run_parse(config, labels)
    else:
        print("No labels data loaded, cannot proceed.")