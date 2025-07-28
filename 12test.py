# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, kurtosis
import re

# 确保 Ksigma 检测器可用
try:
    # 假设 Ksigma 实现位于此路径
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from detector.k_sigma import Ksigma
except ImportError:
    print("FATAL: detector.k_sigma.Ksigma not found. Please ensure the file exists and is in the correct path.")
    sys.exit(1)

tz = pytz.timezone('Asia/Shanghai')  # HKT, UTC+8

def ts_to_date(timestamp_ms):
    if pd.isna(timestamp_ms): return "N/A"
    return datetime.fromtimestamp(timestamp_ms / 1000, tz).strftime('%Y-%m-%d %H:%M:%S')

def time_to_ts(ctime, add_8_hours=False):
    try:
        dt_format = '%Y-%m-%d %H:%M:%S.%f' if '.' in ctime else '%Y-%m-%d %H:%M:%S'
        dt = datetime.strptime(ctime, dt_format)
    except ValueError:
        dt = datetime.strptime(ctime, '%Y-%m-%d')
    
    if add_8_hours:
        dt += timedelta(hours=8)
    
    dt_localized = tz.localize(dt, is_dst=None)
    return int(dt_localized.timestamp()) * 1000

def normalize_kpi_name(kpi):
    return re.sub(r'[^a-z0-9]', '', kpi.lower())

def compute_adaptive_config(metric_data, metric_column, default_config=None):
    if default_config is None:
        default_config = {'k_s': {'k_thr': 1.2, 'std_thr': 0.005, 'win_size': 10}}
    
    config = {'k_s': {}}
    if metric_column not in metric_data.columns or len(metric_data) < 5:
        return default_config
    
    values = metric_data[metric_column].dropna()
    if len(values) < 5: return default_config
        
    with np.errstate(all='ignore'): # 忽略计算警告
        skewness = abs(skew(values, bias=False))
        kurt = kurtosis(values, bias=False)

    config['k_s']['k_thr'] = 0.8 if skewness > 0.7 or kurt > 1.5 else 1.2
    
    mean, std = values.mean(), values.std()
    cv = std / mean if mean != 0 else 0
    if cv < 0.01: std_thr = max(std * 0.15, 0.0005)
    elif cv > 0.3: std_thr = std * 0.08
    else: std_thr = std * 0.03
    config['k_s']['std_thr'] = round(std_thr, 5)
    
    if 'timestamp' in metric_data.columns:
        timestamps = metric_data['timestamp'].dropna().sort_values()
        if len(timestamps) > 1:
            time_diff_ms = timestamps.diff().mean()
            if pd.isna(time_diff_ms) or time_diff_ms <= 0: time_diff_ms = 60000
            win_size = max(5, int((600 * 1000) / time_diff_ms))
            config['k_s']['win_size'] = min(win_size, 12)
        else: config['k_s']['win_size'] = 10
    else: config['k_s']['win_size'] = 10
    
    return config

class MetricEvent:
    def __init__(self, cases, data_dir, dataset='tt', config=None, detector_config=None, adaptive_config=False, score_threshold=0.3):
        self.cases = cases
        self.data_dir = data_dir
        self.dataset = dataset
        self.adaptive_config = adaptive_config
        self.score_threshold = score_threshold # **可配置的分数阈值**
        
        self.metrics = self.get_tt_metric_names(data_dir) if dataset == 'tt' else []
        
        self.config = config or {'minute': 60000, 'MIN_TEST_LENGTH': 3, 'time_margin': 5}
        
        self.metric_configs = {}
        self.detector = Ksigma(detector_config) if not adaptive_config else None
        
        self.res_original = {str(i): [] for i in self.cases.index}
        self.res_offset = {str(i): [] for i in self.cases.index}

        self.stats = {'reads': 0, 'failures': 0, 'anomalies_orig': 0, 'anomalies_off': 0}
    
    def get_tt_metric_names(self, data_dir):
        metric_files = [os.path.join(day, 'metric', f) for day in ['2023-01-29', '2023-01-30'] 
                        if os.path.exists(os.path.join(data_dir, day, 'metric'))
                        for f in os.listdir(os.path.join(data_dir, day, 'metric')) if f.endswith('.csv')]
        print(f"Found {len(metric_files)} metric files. Sample: {metric_files[:3]}")
        return metric_files
    
    def read(self, metric_relative_path):
        filepath = os.path.join(self.data_dir, metric_relative_path)
        if not os.path.exists(filepath): return pd.DataFrame()
        try:
            data = pd.read_csv(filepath)
            if 'TimeStamp' not in data.columns: return pd.DataFrame()
            data['timestamp'] = data['TimeStamp'] * 1000
            if self.adaptive_config:
                metric_prefix = os.path.basename(metric_relative_path)
                for col in ['CpuUsage(m)', 'PodClientLatencyP90(s)', 'PodSuccessRate(%)']:
                    if col in data.columns:
                        self.metric_configs[f"{metric_prefix}_{col}"] = compute_adaptive_config(data, col)
            self.stats['reads'] += 1
            return data
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
    
    def detect_anomalies_for_services(self, metric_data, start_ts_ms, end_ts_ms, case_id, metric_filename):
        anomalies = []
        metric_cols = ['CpuUsage(m)', 'CpuUsageRate(%)', 'MemoryUsage(Mi)', 'MemoryUsageRate(%)',
                       'PodClientLatencyP90(s)', 'PodServerLatencyP90(s)', 'PodSuccessRate(%)']
        
        # 如果没有PodName列，则将整个文件视为一个服务
        if 'PodName' not in metric_data.columns:
            metric_data['PodName'] = metric_filename

        for metric_col in [c for c in metric_cols if c in metric_data.columns]:
            detector = self.detector
            if self.adaptive_config:
                config = self.metric_configs.get(f"{metric_filename}_{metric_col}")
                if config: detector = Ksigma(config)
            if not detector: continue

            for pod_name, service_data in metric_data.groupby('PodName'):
                if len(service_data) < self.config['MIN_TEST_LENGTH']: continue
                
                try:
                    # **深度调试日志**
                    print(f"\n--- Detecting: Case={case_id}, Pod={pod_name}, Metric={metric_col} ---")
                    print(f"Data points: {len(service_data)}, Time range: {ts_to_date(service_data['timestamp'].min())} to {ts_to_date(service_data['timestamp'].max())}")
                    stats_desc = service_data[metric_col].describe()
                    print(f"Stats: mean={stats_desc.get('mean', 0):.4f}, std={stats_desc.get('std', 0):.4f}, min={stats_desc.get('min', 0):.4f}, max={stats_desc.get('max', 0):.4f}")

                    res = detector.detection(service_data, metric_col, start_ts_ms / 1000, end_ts_ms / 1000)
                    
                    # **打印原始检测结果**
                    print(f"Ksigma raw result: {res}")
                    
                    if res and res[0] is True:
                        if res[2] > self.score_threshold:
                            anomaly_ts_ms = int(res[1] * 1000) if res[1] else None
                            normalized_kpi = normalize_kpi_name(metric_col)
                            anomalies.append([anomaly_ts_ms, pod_name, normalized_kpi, res[2]])
                            print(f"  >>> ANOMALY DETECTED! Score {res[2]:.4f} > {self.score_threshold}")
                        else:
                            print(f"  ... Anomaly flag is True, but score {res[2]:.4f} <= threshold {self.score_threshold}. REJECTED.")
                    else:
                        print("  ... No anomaly detected (flag is False or result is empty).")

                except Exception as e:
                    self.stats['failures'] += 1
                    print(f"  *** ERROR during detection for {pod_name}, {metric_col}: {e}")
        return anomalies
    
    def get_metric_events(self):
        print(f"Processing {len(self.metrics)} metrics for {len(self.cases)} cases. Score threshold: {self.score_threshold}")
        margin = self.config['minute'] * self.config['time_margin']
        
        for metric_path in tqdm(self.metrics, desc="Processing metrics"):
            metric_data = self.read(metric_path)
            if metric_data.empty: continue
            
            for case_id, case in self.cases.iterrows():
                try:
                    st_ms, ed_ms = time_to_ts(case['st_time']), time_to_ts(case['ed_time'])
                    d_start, d_end = st_ms - margin, ed_ms + margin
                    
                    win_data = metric_data[(metric_data['timestamp'] >= d_start) & (metric_data['timestamp'] <= d_end)]
                    if not win_data.empty:
                        anom = self.detect_anomalies_for_services(win_data, d_start, d_end, str(case_id), os.path.basename(metric_path))
                        if anom: self.res_original[str(case_id)].extend(anom); self.stats['anomalies_orig'] += len(anom)
                    
                    st_off, ed_off = time_to_ts(case['st_time'], True), time_to_ts(case['ed_time'], True)
                    d_start_off, d_end_off = st_off - margin, ed_off + margin
                    
                    win_data_off = metric_data[(metric_data['timestamp'] >= d_start_off) & (metric_data['timestamp'] <= d_end_off)]
                    if not win_data_off.empty:
                        anom_off = self.detect_anomalies_for_services(win_data_off, d_start_off, d_end_off, str(case_id), os.path.basename(metric_path))
                        if anom_off: self.res_offset[str(case_id)].extend(anom_off); self.stats['anomalies_off'] += len(anom_off)
                except Exception as e:
                    self.stats['failures'] += 1
                    print(f"Error processing case {case_id} for {metric_path}: {e}")
        self.print_stats()
    
    def print_stats(self):
        print("\n=== Detection Statistics Summary ===")
        print(f"Metrics Read: {self.stats['reads']}, Detection Failures: {self.stats['failures']}")
        print(f"Total Anomalies (Original): {self.stats['anomalies_orig']}")
        print(f"Total Anomalies (Offset 8h): {self.stats['anomalies_off']}")
        print(f"Cases w/ Anomalies (Original): {sum(1 for e in self.res_original.values() if e)}/{len(self.cases)}")
        print(f"Cases w/ Anomalies (Offset 8h): {sum(1 for e in self.res_offset.values() if e)}/{len(self.cases)}")

    def save_res(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(f"{save_path}_original.json", 'w') as f: json.dump(self.res_original, f, indent=2)
        with open(f"{save_path}_offset_8h.json", 'w') as f: json.dump(self.res_offset, f, indent=2)
        print(f'Results saved to {save_path}_original.json and _offset_8h.json')

def run_detection(labels, data_dir, config_name, detector_cfg, adaptive, score_thr, save_path):
    print(f"\n{'='*60}\nRunning detection: '{config_name}' | Score Threshold: {score_thr}\n{'='*60}")
    event_processor = MetricEvent(labels, data_dir, 'tt', detector_config=detector_cfg, adaptive_config=adaptive, score_threshold=score_thr)
    event_processor.get_metric_events()
    event_processor.save_res(f"{save_path}_{config_name}")
    return {
        'events_original': sum(len(e) for e in event_processor.res_original.values()),
        'cases_original': sum(1 for e in event_processor.res_original.values() if e),
        'events_offset': sum(len(e) for e in event_processor.res_offset.values()),
        'cases_offset': sum(1 for e in event_processor.res_offset.values() if e)
    }

if __name__ == '__main__':
    data_base_dir = '/home/fuxian/DataSet/tt'
    save_path_base = '/home/fuxian/lixuyang/DiagFusion25+tt/liningshuai/metric_events_tt'
    
    labels_list = [pd.read_csv(os.path.join(data_base_dir, day, 'groundtruth.csv'))
                   for day in ['2023-01-29', '2023-01-30']
                   if os.path.exists(os.path.join(data_base_dir, day, 'groundtruth.csv'))]
    if not labels_list:
        print("No groundtruth files found! Exiting."); sys.exit(1)
    
    labels = pd.concat(labels_list).set_index('index', drop=False)
    print(f"Total loaded: {len(labels)} fault cases from groundtruth.")

    # **扩展测试配置**
    configs_to_run = {
        'adaptive': {'cfg': None, 'adaptive': True, 'score': 0.3},
        'default_strict': {'cfg': {'k_s': {'k_thr': 1.0, 'std_thr': 0.01, 'win_size': 10}}, 'adaptive': False, 'score': 0.3},
        'default_relaxed': {'cfg': {'k_s': {'k_thr': 0.8, 'std_thr': 0.001, 'win_size': 12}}, 'adaptive': False, 'score': 0.1},
    }
    
    summary = {}
    for name, p in configs_to_run.items():
        summary[name] = run_detection(labels, data_base_dir, name, p['cfg'], p['adaptive'], p['score'], save_path_base)
    
    print(f"\n{'='*80}\nFINAL COMPARISON SUMMARY\n{'='*80}")
    print(f"Total fault cases: {len(labels)}")
    print(f"\n{'Config':<20} {'Cases (Orig)':<15} {'Events (Orig)':<15} {'Cases (Offset)':<15} {'Events (Offset)':<15}")
    print("-" * 80)
    for name, res in summary.items():
        print(f"{name:<20} {str(res['cases_original'])+'/'+str(len(labels)):<15} {res['events_original']:<15} {str(res['cases_offset'])+'/'+str(len(labels)):<15} {res['events_offset']:<15}")