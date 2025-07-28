# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')))
import json
import pandas as pd
from detector.k_sigma import Ksigma
from datetime import datetime, timedelta
import pytz
import time
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, kurtosis

tz = pytz.timezone('Asia/Shanghai')

def ts_to_date(timestamp):
    try:
        return datetime.fromtimestamp(timestamp, tz).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return datetime.fromtimestamp(timestamp//1000, tz).strftime('%Y-%m-%d %H:%M:%S')

def time_to_ts(ctime, add_8_hours=False):
    """
    将日期时间字符串转换为 UTC 时间戳（毫秒），可选择是否加8小时偏移
    
    Args:
        ctime: 日期时间字符串，如 '2023-01-29 08:43:04.000000'
        add_8_hours: 是否在时间上加8小时
    
    Returns:
        int: UTC 时间戳（毫秒）
    """
    try:
        if '.' in ctime:
            timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f')
        else:
            timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S')
    except:
        timeArray = time.strptime(ctime, '%Y-%m-%d')
    
    # 将时间视为 UTC+8
    dt = datetime(*timeArray[:6])
    if add_8_hours:
        dt = dt + timedelta(hours=8)
    dt = tz.localize(dt).astimezone(pytz.UTC)
    return int(dt.timestamp()) * 1000

def compute_adaptive_config(metric_data, default_config=None):
    """
    根据 TT 数据集优化自适应配置，适配 60s 采样和 600s 异常
    """
    if default_config is None:
        default_config = {'k_s': {'k_thr': 1.2, 'std_thr': 0.005, 'win_size': 10}}
    
    config = {'k_s': {}}
    
    if 'value' not in metric_data.columns or len(metric_data) < 5:
        print("Warning: Insufficient data for adaptive config, using default")
        return default_config
    
    values = metric_data['value'].dropna()
    
    # 1. 计算 k_thr：适配 TT 数据集的高波动 cpuusage（如 1.307-6.092）
    skewness = abs(skew(values, bias=False))
    kurt = kurtosis(values, bias=False)
    # 调整：k_thr 更敏感以捕获高波动
    if skewness > 0.7 or kurt > 1.5:
        k_thr = 0.8
    else:
        k_thr = 1.2
    config['k_s']['k_thr'] = k_thr
    
    # 2. 计算 std_thr：适配 TT 数据集的值范围
    mean = values.mean()
    std = values.std()
    cv = std / mean if mean != 0 else 0
    # 调整：增加 std_thr 以适配高波动
    if cv < 0.01:
        std_thr = max(std * 0.15, 0.0005)
    elif cv > 0.3:
        std_thr = std * 0.08
    else:
        std_thr = std * 0.03
    config['k_s']['std_thr'] = round(std_thr, 5)
    
    # 3. 计算 win_size：适配 60s 采样和 600s 异常
    if 'timestamp' in metric_data.columns:
        timestamps = metric_data['timestamp'].dropna()
        if len(timestamps) > 1:
            time_diff = (timestamps.max() - timestamps.min()) / (len(timestamps) - 1)
            typical_anomaly_duration = 600
            win_size = max(5, int(typical_anomaly_duration / time_diff))  # 覆盖异常时长
            win_size = min(win_size, 12)  # 上限 12 点（720s）
        else:
            win_size = 10
    else:
        win_size = 10
    config['k_s']['win_size'] = win_size
    
    print(f"Adaptive config for metric: k_thr={k_thr}, std_thr={std_thr}, win_size={win_size}")
    return config

class MetricEvent:
    def __init__(self, cases, metric_path, data_dir, dataset='tt', config=None, detector_config=None, adaptive_config=False):
        self.cases = cases
        self.debug = True
        self.debug_count = 0
        self.detailed_debug = True  # 启用详细调试
        self.adaptive_config = adaptive_config
        
        if dataset == 'tt':
            self.metrics = self.get_tt_metric_names(data_dir)
        else:
            raise Exception(f'Unknown dataset {dataset}')
        
        self.data_dir = data_dir
        self.dataset = dataset
        
        if config is None:
            config = {'minute': 60000, 'MIN_TEST_LENGTH': 3}
        self.config = config
        
        self.metric_configs = {}
        
        if not adaptive_config:
            if detector_config is None:
                detector_config = {'k_s': {'k_thr': 1.2, 'std_thr': 0.005, 'win_size': 10}}
            self.detector = Ksigma(detector_config)
            self.detector_config = detector_config
        else:
            self.detector = None
            self.detector_config = None
        
        self.res_original = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))
        self.res_offset = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))
        self.stats_original = {
            'total_metrics': 0, 'successful_reads': 0, 'total_detections': 0,
            'cases_with_data': 0, 'cases_without_data': 0, 'detection_failures': 0
        }
        self.stats_offset = {
            'total_metrics': 0, 'successful_reads': 0, 'total_detections': 0,
            'cases_with_data': 0, 'cases_without_data': 0, 'detection_failures': 0
        }
    
    def get_tt_metric_names(self, data_dir):
        metric_files = []
        tt_days = ['2023-01-29', '2023-01-30']
        
        for day in tt_days:
            metric_dir = os.path.join(data_dir, day, 'metric')
            if os.path.exists(metric_dir):
                for csv_file in os.listdir(metric_dir):
                    if csv_file.endswith('.csv'):
                        metric_files.append(f"{day}_{csv_file}")
            else:
                print(f"Warning: Metric directory not found: {metric_dir}")
        
        print(f"Found {len(metric_files)} metric files for TT dataset")
        if self.debug and metric_files:
            print(f"Sample metric files: {metric_files[:5]}")
        return metric_files
    
    def read(self, metric):
        parts = metric.split('_', 1)
        if len(parts) == 2:
            date, filename = parts
            filepath = os.path.join(self.data_dir, date, 'metric', filename)
        else:
            filepath = os.path.join(self.data_dir, metric)
        
        if not os.path.exists(filepath):
            if self.debug:
                print(f"Warning: File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(filepath)
            if self.debug and self.debug_count < 5:
                print(f"\n=== DEBUG: Metric file {self.debug_count + 1} ===")
                print(f"File: {filepath}")
                print(f"Shape: {data.shape}")
                print(f"Columns: {data.columns.tolist()}")
                if 'timestamp' in data.columns:
                    print(f"Timestamp range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                    print(f"Readable timestamp range (UTC): "
                          f"{datetime.fromtimestamp(data['timestamp'].min(), pytz.UTC)} to "
                          f"{datetime.fromtimestamp(data['timestamp'].max(), pytz.UTC)}")
                    print(f"Readable timestamp range (UTC+8): "
                          f"{datetime.fromtimestamp(data['timestamp'].min(), tz)} to "
                          f"{datetime.fromtimestamp(data['timestamp'].max(), tz)}")
                if 'value' in data.columns:
                    print(f"Value stats: min={data['value'].min():.4f}, max={data['value'].max():.4f}")
                    print(f"Value mean={data['value'].mean():.4f}, std={data['value'].std():.4f}")
                    value_range = data['value'].max() - data['value'].min()
                    value_cv = data['value'].std() / data['value'].mean() if data['value'].mean() != 0 else 0
                    print(f"Value range: {value_range:.4f}, CV: {value_cv:.4f}")
                if 'cmdb_id' in data.columns:
                    print(f"Services: {data['cmdb_id'].nunique()}, sample: {list(data['cmdb_id'].unique()[:3])}")
                self.debug_count += 1
            
            if 'timestamp' in data.columns:
                data.index = [ts_to_date(ts) for ts in data['timestamp']]
            else:
                print(f"Warning: No timestamp column found in {filepath}")
                return pd.DataFrame()
            
            if self.adaptive_config:
                self.metric_configs[metric] = compute_adaptive_config(data)
            
            return data
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
    
    def detect_anomalies_for_services(self, metric_data, start_ts, end_ts, kpi_name, case_id, metric=None, stats=None):
        anomalies = []
        
        if 'cmdb_id' not in metric_data.columns:
            if self.detailed_debug:
                print(f"  No cmdb_id column in data")
            return anomalies
        
        services_processed = 0
        services_with_enough_data = 0
        services_with_anomalies = 0
        
        if self.adaptive_config and metric in self.metric_configs:
            detector = Ksigma(self.metric_configs[metric])
        else:
            detector = self.detector
        
        for cmdb_id in metric_data['cmdb_id'].unique():
            services_processed += 1
            service_data = metric_data[metric_data['cmdb_id'] == cmdb_id].copy()
            
            if len(service_data) < 3:
                if self.detailed_debug:
                    print(f"    Service {cmdb_id}: insufficient data ({len(service_data)} points)")
                continue
            
            services_with_enough_data += 1
            
            try:
                if 'value' in service_data.columns:
                    values = service_data['value']
                    value_mean = values.mean()
                    value_std = values.std()
                    value_cv = value_std / value_mean if value_mean != 0 else 0
                    if self.detailed_debug:
                        print(f"    Service {cmdb_id}: {len(service_data)} points, "
                              f"mean={value_mean:.4f}, std={value_std:.4f}, cv={value_cv:.4f}")
                
                res = detector.detection(service_data, 'value', start_ts/1000, end_ts/1000)
                
                if res[0] is True and res[2] > 0.3:
                    services_with_anomalies += 1
                    anomaly_ts_ms = int(res[1] * 1000) if res[1] else None
                    anomalies.append([anomaly_ts_ms, cmdb_id, kpi_name, res[2]])
                    if self.debug:
                        print(f"    ANOMALY! Service: {cmdb_id}, KPI: {kpi_name}, Score: {res[2]:.4f}")
                elif self.detailed_debug:
                    print(f"    Service {cmdb_id}: no anomaly detected, score: {res[2]:.4f}")
                        
            except Exception as e:
                stats['detection_failures'] += 1
                if self.debug:
                    print(f"    Error detecting anomaly for service {cmdb_id}: {e}")
                continue
        
        if self.debug:
            print(f"  Services summary: {services_processed} total, "
                  f"{services_with_enough_data} with enough data, "
                  f"{services_with_anomalies} with anomalies")
                
        return anomalies
    
    def get_metric_events(self):
        print(f"Processing {len(self.metrics)} metrics for {len(self.cases)} cases...")
        if self.adaptive_config:
            print("Using adaptive configuration per metric")
        else:
            print(f"Using detector config: {self.detector_config}")
        
        self.stats_original['total_metrics'] = len(self.metrics)
        self.stats_offset['total_metrics'] = len(self.metrics)
        
        if self.debug:
            print(f"\n=== DEBUG: Fault cases analysis ===")
            print(f"Total cases: {len(self.cases)}")
            for i, (case_id, case) in enumerate(self.cases.head(5).iterrows()):
                start_ts = time_to_ts(case['st_time'])
                end_ts = time_to_ts(case['ed_time'])
                start_ts_offset = time_to_ts(case['st_time'], add_8_hours=True)
                end_ts_offset = time_to_ts(case['ed_time'], add_8_hours=True)
                detection_start = start_ts - self.config['minute'] * 15
                detection_end = end_ts + self.config['minute'] * 5
                detection_start_offset = start_ts_offset - self.config['minute'] * 15
                detection_end_offset = end_ts_offset + self.config['minute'] * 5
                print(f"  Case {case_id}:")
                print(f"    Original: {case['st_time']} to {case['ed_time']}")
                print(f"    Detection window: {ts_to_date(detection_start/1000)} to {ts_to_date(detection_end/1000)}")
                print(f"    Offset: {ts_to_date(start_ts_offset/1000)} to {ts_to_date(end_ts_offset/1000)}")
                print(f"    Detection window (offset): {ts_to_date(detection_start_offset/1000)} to {ts_to_date(detection_end_offset/1000)}")
        
        for metric in tqdm(self.metrics, desc="Processing metrics"):
            metric_data = self.read(metric)
            if metric_data.empty:
                continue
                
            self.stats_original['successful_reads'] += 1
            self.stats_offset['successful_reads'] += 1
            
            if 'kpi_name' in metric_data.columns:
                kpi_name = metric_data['kpi_name'].iloc[0]
            else:
                parts = metric.split('_', 1)
                if len(parts) == 2:
                    kpi_name = parts[1].replace('.csv', '')
                else:
                    kpi_name = metric.replace('.csv', '')
            
            case_events_original = 0
            case_events_offset = 0
            
            for case_id, case in self.cases.iterrows():
                try:
                    self.stats_original['total_detections'] += 1
                    self.stats_offset['total_detections'] += 1
                    
                    # Original timestamps
                    start_ts = time_to_ts(case['st_time']) - self.config['minute'] * 15
                    end_ts = time_to_ts(case['ed_time']) + self.config['minute'] * 5
                    # Offset timestamps
                    start_ts_offset = time_to_ts(case['st_time'], add_8_hours=True) - self.config['minute'] * 15
                    end_ts_offset = time_to_ts(case['ed_time'], add_8_hours=True) + self.config['minute'] * 5
                    
                    # Original window
                    window_start = start_ts / 1000
                    window_end = end_ts / 1000
                    window_data = metric_data[(metric_data['timestamp'] >= window_start) & 
                                            (metric_data['timestamp'] <= window_end)]
                    
                    if len(window_data) > 0:
                        self.stats_original['cases_with_data'] += 1
                        if self.detailed_debug and case_id <= 5:
                            print(f"\n=== DEBUG: Case {case_id}, Metric {kpi_name} (Original) ===")
                            print(f"  Window data: {len(window_data)} points")
                            if 'cmdb_id' in window_data.columns:
                                print(f"  Services in window: {window_data['cmdb_id'].nunique()}")
                                print(f"  Sample cmdb_id: {list(window_data['cmdb_id'].unique()[:3])}")
                        anomalies = self.detect_anomalies_for_services(
                            window_data, start_ts, end_ts, kpi_name, case_id, metric, self.stats_original
                        )
                        for anomaly in anomalies:
                            self.res_original[case_id].append(anomaly)
                            case_events_original += 1
                    else:
                        self.stats_original['cases_without_data'] += 1
                        if self.detailed_debug and case_id <= 5:
                            print(f"  Case {case_id}, Metric {kpi_name} (Original): No data in window")
                            print(f"  Window: {ts_to_date(window_start)} to {ts_to_date(window_end)}")
                    
                    # Offset window
                    window_start_offset = start_ts_offset / 1000
                    window_end_offset = end_ts_offset / 1000
                    window_data_offset = metric_data[(metric_data['timestamp'] >= window_start_offset) & 
                                                   (metric_data['timestamp'] <= window_end_offset)]
                    
                    if len(window_data_offset) > 0:
                        self.stats_offset['cases_with_data'] += 1
                        if self.detailed_debug and case_id <= 5:
                            print(f"\n=== DEBUG: Case {case_id}, Metric {kpi_name} (Offset) ===")
                            print(f"  Window data: {len(window_data_offset)} points")
                            if 'cmdb_id' in window_data_offset.columns:
                                print(f"  Services in window: {window_data_offset['cmdb_id'].nunique()}")
                                print(f"  Sample cmdb_id: {list(window_data_offset['cmdb_id'].unique()[:3])}")
                        anomalies = self.detect_anomalies_for_services(
                            window_data_offset, start_ts_offset, end_ts_offset, kpi_name, case_id, metric, self.stats_offset
                        )
                        for anomaly in anomalies:
                            self.res_offset[case_id].append(anomaly)
                            case_events_offset += 1
                    else:
                        self.stats_offset['cases_without_data'] += 1
                        if self.detailed_debug and case_id <= 5:
                            print(f"  Case {case_id}, Metric {kpi_name} (Offset): No data in window")
                            print(f"  Window: {ts_to_date(window_start_offset)} to {ts_to_date(window_end_offset)}")
                        
                except Exception as e:
                    self.stats_original['detection_failures'] += 1
                    self.stats_offset['detection_failures'] += 1
                    if self.debug:
                        print(f"Error processing case {case_id} for metric {metric}: {e}")
                    continue
            
            if case_events_original > 0:
                print(f"Metric {kpi_name} (Original): Found {case_events_original} anomalous events")
            if case_events_offset > 0:
                print(f"Metric {kpi_name} (Offset): Found {case_events_offset} anomalous events")
        
        self.print_stats()
    
    def print_stats(self):
        for label, stats in [("Original", self.stats_original), ("Offset 8h", self.stats_offset)]:
            print(f"\n=== Detection Statistics ({label}) ===")
            print(f"Total metrics: {stats['total_metrics']}")
            print(f"Successfully read: {stats['successful_reads']}")
            print(f"Total detections attempted: {stats['total_detections']}")
            print(f"Cases with data: {stats['cases_with_data']}")
            print(f"Cases without data: {stats['cases_without_data']}")
            print(f"Detection failures: {stats['detection_failures']}")
            
            total_anomalies = sum(len(events) for events in (self.res_original if label == "Original" else self.res_offset).values())
            cases_with_anomalies = sum(1 for events in (self.res_original if label == "Original" else self.res_offset).values() if len(events) > 0)
            
            print(f"Total anomalies detected: {total_anomalies}")
            print(f"Cases with anomalies: {cases_with_anomalies}/{len(self.cases)}")
    
    def save_res(self, savepath_original, savepath_offset):
        os.makedirs(os.path.dirname(savepath_original), exist_ok=True)
        os.makedirs(os.path.dirname(savepath_offset), exist_ok=True)
        with open(savepath_original, 'w') as f:
            json.dump(self.res_original, f, indent=2)
        print(f'Original results saved successfully to: {savepath_original}')
        with open(savepath_offset, 'w') as f:
            json.dump(self.res_offset, f, indent=2)
        print(f'Offset results saved successfully to: {savepath_offset}')

def run_detection_with_config(labels, data_base_dir, detector_config, config_name, adaptive_config=False, save_path_base=None):
    print(f"\n{'='*60}")
    print(f"Running detection with {config_name} configuration")
    if adaptive_config:
        print("Using adaptive configuration per metric")
    else:
        print(f"Config: {detector_config}")
    print(f"{'='*60}")
    
    metric_event = MetricEvent(
        labels, None, data_base_dir, 'tt',
        detector_config=detector_config if not adaptive_config else None,
        adaptive_config=adaptive_config
    )
    metric_event.get_metric_events()
    
    save_path_original = f'{save_path_base}_original.json' if save_path_base else f'/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/metric_events_tt_{config_name}_original.json'
    save_path_offset = f'{save_path_base}_offset_8h.json' if save_path_base else f'/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/metric_events_tt_{config_name}_offset_8h.json'
    metric_event.save_res(save_path_original, save_path_offset)
    
    for label, res in [("Original", metric_event.res_original), ("Offset 8h", metric_event.res_offset)]:
        total_events = sum(len(events) for events in res.values())
        cases_with_events = sum(1 for events in res.values() if len(events) > 0)
        
        print(f"\n=== {config_name} Results Summary ({label}) ===")
        print(f"Total fault cases: {len(labels)}")
        print(f"Cases with detected events: {cases_with_events}")
        print(f"Total anomalous metric events: {total_events}")
        print(f"Results saved to: {save_path_original if label == 'Original' else save_path_offset}")
        
        print(f"\n=== {config_name} Case-by-Case Results ({label}) ===")
        for case_id, events in res.items():
            if len(events) > 0:
                print(f"Case {case_id}: {len(events)} events")
                for event in events[:3]:
                    if len(event) >= 4:
                        timestamp, cmdb_id, kpi_name, score = event
                        readable_time = ts_to_date(timestamp/1000) if timestamp else "Unknown"
                        print(f"  - {readable_time}: {cmdb_id} - {kpi_name} (score: {score:.3f})")
                if len(events) > 3:
                    print(f"  ... and {len(events) - 3} more events")
            else:
                print(f"Case {case_id}: No events detected")
    
    return metric_event

if __name__ == '__main__':
    data_base_dir = '/home/fuxian/DataSet/NewDataset/tt'
    tt_days = ['2023-01-29', '2023-01-30']
    
    # 读取 groundtruth 文件
    labels = []
    print("=== Loading TT groundtruth files ===")
    for day_str in tt_days:
        label_path = os.path.join(data_base_dir, day_str, 'groundtruth.csv')
        if os.path.exists(label_path):
            try:
                df = pd.read_csv(label_path)
                print(f"Loaded {len(df)} cases from {label_path}")
                labels.append(df)
            except Exception as e:
                print(f"Error loading {label_path}: {e}")
        else:
            print(f"Missing: {label_path}")
    
    if not labels:
        print("No groundtruth files found!")
        sys.exit(1)
    
    labels = pd.concat(labels, ignore_index=True)
    print(f"Total loaded: {len(labels)} fault cases")
    
    # 定义测试配置，调整以适配 TT 数据集
    configs = {
        'adaptive': None,
        'default': {
            'k_s': {
                'k_thr': 1.0,  # 更敏感以适配高波动
                'std_thr': 0.01,  # 适配 TT 的 cpuusage 范围
                'win_size': 10  # 保持 600s 异常窗口
            }
        }
    }
    
    # 运行测试
    results = {}
    for config_name, detector_config in configs.items():
        metric_event = run_detection_with_config(
            labels, data_base_dir, detector_config,
            config_name,
            adaptive_config=(config_name == 'adaptive')
        )
        results[config_name] = {
            'events_original': sum(len(events) for events in metric_event.res_original.values()),
            'cases_original': sum(1 for events in metric_event.res_original.values() if len(events) > 0),
            'events_offset': sum(len(events) for events in metric_event.res_offset.values()),
            'cases_offset': sum(1 for events in metric_event.res_offset.values() if len(events) > 0)
        }
    
    # 打印最终对比
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: TT (2023-01-29, 2023-01-30)")
    print(f"Total fault cases: {len(labels)}")
    print(f"\n{'Config':<15} {'Cases (Original)':<15} {'Events (Original)':<18} {'Cases (Offset)':<15} {'Events (Offset)':<15}")
    print("-" * 80)
    
    for config_name in configs:
        cases_original = results[config_name]['cases_original']
        events_original = results[config_name]['events_original']
        cases_offset = results[config_name]['cases_offset']
        events_offset = results[config_name]['events_offset']
        print(f"{config_name:<15} {cases_original:<15} {events_original:<18} {cases_offset:<15} {events_offset:<15}")
    
    print(f"\nResults saved:")
    for config_name in configs.keys():
        print(f"  - /home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/metric_events_tt_{config_name}_original.json")
        print(f"  - /home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/metric_events_tt_{config_name}_offset_8h.json")