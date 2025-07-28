import os
import math
import json
import pandas as pd
from collections import defaultdict

def load_groundtruth(groundtruth_path):
    """加载 groundtruth.csv 文件"""
    if os.path.exists(groundtruth_path):
        labels = pd.read_csv(groundtruth_path)
        print(f"Loaded {len(labels)} cases from {groundtruth_path}")
        labels['index'] = labels['index'].astype(str)
        labels.set_index('index', inplace=True)
        return labels
    else:
        print(f"Error: File not found: {groundtruth_path}")
        return pd.DataFrame()

def load_json_data(file_path, data_type):
    """加载 JSON 文件（trace、metric 或 log）"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {data_type} data from {file_path}, {len(data)} cases")
        # 确保 case_id 为字符串
        data = {str(k): v for k, v in data.items()}
        return data
    else:
        print(f"Warning: File not found: {file_path}")
        return None

def normalize_service_name(name, instance_names):
    """从实例名提取服务名"""
    if not isinstance(name, str):
        return None
    for service in instance_names:
        if name.startswith(service):
            return service
    return None

def metric_trace_log_parse(trace, metric, logs, labels, save_path, nodes):
    """三模态数据融合处理"""
    # 移除 metric 中的 np.inf 值
    if metric is not None:
        for k, v in metric.items():
            metric[k] = [x for x in v if len(x) >= 4 and not math.isinf(x[3])]

    # 确保 logs 格式与 labels 索引对齐
    log = None
    if logs is not None:
        log = {str(x): logs.get(str(x), []) for x in labels.index}
    else:
        log = {str(x): [] for x in labels.index}
        print("WARNING: No log data provided, using empty lists")

    # 获取所有服务名
    instance_names = nodes.split()
    print(f"DEBUG: instance_names: {instance_names}")

    # 获取异常服务和异常类型
    anomaly_instances = labels['service'].values
    anomaly_type = labels['anomaly_type'].values
    print(f"DEBUG: anomaly_instances: {anomaly_instances[:5]}")
    print(f"DEBUG: anomaly_type: {anomaly_type[:5]}")

    # 初始化输出字典
    demo_metric = defaultdict(lambda: defaultdict(list))

    # 遍历每个 case_id
    for k in labels.index:
        case_id = str(k)
        print(f"DEBUG: Processing case_id {case_id}")

        # 获取当前 case 的异常服务和异常类型
        anomaly_instance_base = anomaly_instances[k]
        if anomaly_instance_base not in instance_names:
            print(f"WARNING: anomaly_instance_base '{anomaly_instance_base}' not in instance_names for case {case_id}")
            continue

        # 生成所有服务的键：异常服务使用 anomaly_type，其他服务使用 [normal]
        inner_dict_key = [(x, anomaly_type[k] if x == anomaly_instance_base else "[normal]") for x in instance_names]
        print(f"DEBUG: inner_dict_key for case {case_id}: {inner_dict_key}")

        # 初始化 demo_metric 的键
        for key in inner_dict_key:
            demo_metric[case_id][key] = []

        # 处理 metric 数据
        if metric is not None and case_id in metric:
            for inner_key in inner_dict_key:
                instance_base = inner_key[0]
                matching_metrics = []
                for y in metric[case_id]:
                    if len(y) >= 4:
                        cmdb_id = y[1]
                        cmdb_id_base = normalize_service_name(cmdb_id, instance_names)
                        if cmdb_id_base == instance_base:
                            event_name = f"{cmdb_id}_{y[2]}_{'+' if y[3] > 0 else '-'}"
                            matching_metrics.append([str(y[0]), event_name])
                demo_metric[case_id][inner_key].extend(matching_metrics)
                print(f"DEBUG: Case {case_id}, Service {instance_base}: {len(matching_metrics)} metrics added")

        # 处理 trace 数据
        if trace is not None and case_id in trace:
            for inner_key in inner_dict_key:
                instance_base = inner_key[0]
                matching_traces = []
                for y in trace[case_id]:
                    if len(y) >= 3 and not math.isinf(y[3] if len(y) > 3 else float('inf')):
                        src_base = normalize_service_name(y[1], instance_names)
                        dst_base = normalize_service_name(y[2], instance_names)
                        if instance_base in (src_base, dst_base):
                            trace_name = f"{y[1]}_{y[2]}"
                            matching_traces.append([str(y[0]), trace_name])
                demo_metric[case_id][inner_key].extend(matching_traces)
                print(f"DEBUG: Case {case_id}, Service {instance_base}: {len(matching_traces)} traces added")

        # 处理 log 数据
        if log is not None and case_id in log:
            for inner_key in inner_dict_key:
                instance_base = inner_key[0]
                matching_logs = []
                for y in log[case_id]:
                    if len(y) >= 3:
                        service_name = normalize_service_name(y[1], instance_names)
                        if service_name == instance_base:
                            matching_logs.append([str(y[0]), y[2]])
                demo_metric[case_id][inner_key].extend(matching_logs)
                print(f"DEBUG: Case {case_id}, Service {instance_base}: {len(matching_logs)} logs added")

    # 转换为 JSON 可序列化的格式
    demo_metric_str_keys = {}
    for case_id, inner_dict in demo_metric.items():
        demo_metric_str_keys[case_id] = {}
        for key_tuple, value in inner_dict.items():
            key_str = str(key_tuple)
            if key_tuple[0] not in instance_names:
                print(f"WARNING: Invalid service name {key_tuple[0]} in case {case_id}, skipping")
                continue
            demo_metric_str_keys[case_id][key_str] = value

    # 保存结果到 JSON 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(demo_metric_str_keys, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {save_path}")

    return demo_metric_str_keys

def main():
    """主函数，加载文件并调用三模态融合函数"""
    # 文件路径
    groundtruth_path = "allgroundtruth.csv"
    log_path = "/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/8.简化三元组_extracted_logs.json"
    metric_path = "/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/metric_events_tt_default_offset_8h.json"
    trace_path = "/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/metric_events_tt_default_original.json"
    save_path = "三模态_tt.json"

    # 服务节点列表
    nodes = (
        "ts-auth-service ts-basic-service ts-cancel-service ts-contacts-service ts-delivery-service "
        "ts-execute-service ts-food-service ts-gateway-service ts-inside-payment-service ts-order-other-service "
        "ts-order-service ts-payment-service ts-preserve-other-service ts-preserve-service ts-price-service "
        "ts-route-service ts-seat-service ts-security-service ts-station-food-service ts-station-service "
        "ts-train-food-service ts-train-service ts-travel-service ts-travel2-service ts-user-service "
        "ts-verification-code-service"
    )

    # 加载数据
    labels = load_groundtruth(groundtruth_path)
    if labels.empty:
        print("Error: Failed to load groundtruth data")
        return

    trace = load_json_data(trace_path, "trace")
    metric = load_json_data(metric_path, "metric")
    logs = load_json_data(log_path, "log")

    # 调用三模态融合函数
    demo_metric_str_keys = metric_trace_log_parse(trace, metric, logs, labels, save_path, nodes)
    print(f"Processing complete. Output JSON contains {len(demo_metric_str_keys)} cases.")

if __name__ == "__main__":
    main()