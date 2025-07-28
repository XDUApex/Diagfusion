
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import logging
from datetime import datetime
import pytz
import time
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, kurtosis

# 假设 Ksigma 检测器已定义
from detector.k_sigma import Ksigma

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置时区
tz = pytz.timezone('Asia/Shanghai')

def ts_to_date(timestamp):
    """将时间戳（秒）转换为可读日期字符串"""
    try:
        return datetime.fromtimestamp(timestamp, tz).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return datetime.fromtimestamp(timestamp // 1000, tz).strftime('%Y-%m-%d %H:%M:%S')

def time_to_ts(ctime):
    """将日期时间字符串或 pandas.Timestamp 转换为时间戳（毫秒）"""
    if isinstance(ctime, pd.Timestamp):
        logger.debug(f"Converting pandas.Timestamp: {ctime}")
        return int(ctime.timestamp() * 1000)  # 直接转换 Timestamp 为毫秒
    elif isinstance(ctime, str):
        logger.debug(f"Converting string: {ctime}")
        try:
            if '.' in ctime:
                time_array = time.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f')
            else:
                time_array = time.strptime(ctime, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            time_array = time.strptime(ctime, '%Y-%m-%d')
        dt = datetime(*time_array[:6])
        return int(dt.timestamp() * 1000)
    else:
        logger.error(f"Invalid input type for time_to_ts: {type(ctime)}")
        raise TypeError(f"time_to_ts expects str or pandas.Timestamp, got {type(ctime)}")

def compute_adaptive_config(metric_data, default_config=None):
    """根据数据集优化自适应配置，适配 60s 采样和 600s 异常"""
    if default_config is None:
        default_config = {'k_s': {'k_thr': 1.0, 'std_thr': 0.005, 'win_size': 8}}
    
    config = {'k_s': {}}
    
    if 'value' not in metric_data.columns or len(metric_data) < 5:
        logger.warning("Insufficient data for adaptive config, using default")
        return default_config
    
    values = metric_data['value'].dropna()
    skewness = abs(skew(values, bias=False))
    kurt = kurtosis(values, bias=False)
    mean = values.mean()
    std = values.std()
    cv = std / mean if mean != 0 else 0
    
    logger.info(f"Data stats: skew={skewness:.4f}, kurtosis={kurt:.4f}, cv={cv:.4f}")
    
    # 降低 k_thr 阈值以提高敏感性
    if skewness > 0.5 or kurt > 1.0:
        k_thr = 0.7
    else:
        k_thr = 1.0
    config['k_s']['k_thr'] = k_thr
    
    # 调整 std_thr 计算逻辑
    if cv < 0.01:
        std_thr = max(std * 0.1, 0.001)
    elif cv > 0.3:
        std_thr = std * 0.05
    else:
        std_thr = std * 0.03
    config['k_s']['std_thr'] = round(std_thr, 5)
    
    # 调整窗口大小
    if 'timestamp' in metric_data.columns:
        timestamps = metric_data['timestamp'].dropna()
        if len(timestamps) > 1:
            time_diff = (timestamps.max() - timestamps.min()) / (len(timestamps) - 1)
            typical_anomaly_duration = 600
            win_size = max(5, int(typical_anomaly_duration / time_diff))
            win_size = min(win_size, 10)
        else:
            win_size = 8
    else:
        win_size = 8
    config['k_s']['win_size'] = win_size
    
    logger.info(f"Adaptive config: k_thr={k_thr}, std_thr={std_thr}, win_size={win_size}")
    return config

class DataLoader:
    def __init__(self, data_dir, debug=False):
        self.data_dir = data_dir
        self.debug = debug
        self.debug_count = 0

    def get_tt_metric_names(self):
        """获取 metric 文件列表，按日期分组"""
        metric_files = {}
        tt_days = ['2023-01-29', '2023-01-30']
        for day in tt_days:
            metric_dir = os.path.join(self.data_dir, day, 'metric')
            if os.path.exists(metric_dir):
                metric_files[day] = [f"{day}_{f}" for f in os.listdir(metric_dir) if f.endswith('.csv')]
                if self.debug and self.debug_count < 5:
                    logger.info(f"Found metric files for {day}: {metric_files[day][:5]}")
                    self.debug_count += 1
        return metric_files

    def read(self, file_path, metric_name):
        """读取 CSV 文件并处理时间戳"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        data = pd.read_csv(file_path)
        if data.empty:
            logger.warning(f"Empty file: {file_path}")
            return pd.DataFrame()
        
        # 选择 value 列
        if 'PodClientLatencyP90(s)' in data.columns:
            data['value'] = data['PodClientLatencyP90(s)']
        elif 'CpuUsage(m)' in data.columns:
            data['value'] = data['CpuUsage(m)']
        elif 'MemoryUsage(Mi)' in data.columns:
            data['value'] = data['MemoryUsage(Mi)']
        else:
            logger.warning(f"No suitable value column found in {file_path}")
            return pd.DataFrame()
        
        # 处理时间戳
        if 'TimeStamp' in data.columns and pd.api.types.is_numeric_dtype(data['TimeStamp']):
            timestamp_col = 'TimeStamp'
            data['timestamp'] = data[timestamp_col] * 1000  # 秒转毫秒
            if self.debug and self.debug_count < 5:
                logger.info(f"Raw TimeStamp (first 5) for {file_path}: {data[timestamp_col].head().tolist()}")
                logger.info(f"Converted timestamp (first 5): {data['timestamp'].head().tolist()}")
                self.debug_count += 1
        elif 'Time' in data.columns:
            cleaned_time = data['Time'].str.replace(r' \+0000 UTC m=\+.*', '', regex=True)
            data['timestamp'] = pd.to_datetime(cleaned_time, errors='coerce').astype('int64') // 10**6
            if data['timestamp'].isna().any():
                logger.warning(f"Some Time values could not be parsed in {file_path}")
                return pd.DataFrame()
            if self.debug and self.debug_count < 5:
                logger.info(f"Raw Time (first 5) for {file_path}: {data['Time'].head().tolist()}")
                logger.info(f"Converted timestamp (first 5): {data['timestamp'].head().tolist()}")
                self.debug_count += 1
        else:
            logger.warning(f"No valid timestamp column found in {file_path}")
            return pd.DataFrame()
        
        data.index = [ts_to_date(ts / 1000) for ts in data['timestamp']]
        if self.debug and self.debug_count < 5:
            logger.info(f"Timestamp range for {metric_name}: {ts_to_date(data['timestamp'].min() / 1000)} to {ts_to_date(data['timestamp'].max() / 1000)}")
            logger.info(f"Value stats for {metric_name}: min={data['value'].min():.4f}, max={data['value'].max():.4f}, mean={data['value'].mean():.4f}, std={data['value'].std():.4f}")
            if 'PodName' in data.columns:
                logger.info(f"Services: {data['PodName'].nunique()}, sample: {list(data['PodName'].unique()[:3])}")
            self.debug_count += 1
        
        return data

    def load_groundtruth(self):
        """加载 groundtruth 文件"""
        gt_files = []
        for day in ['2023-01-29', '2023-01-30']:
            gt_path = os.path.join(self.data_dir, day, 'groundtruth.csv')
            if os.path.exists(gt_path):
                gt_files.append(gt_path)
        
        all_data = []
        for gt_file in gt_files:
            data = pd.read_csv(gt_file)
            data['st_time'] = pd.to_datetime(data['st_time'])
            data['ed_time'] = pd.to_datetime(data['ed_time'])
            all_data.append(data)
            logger.info(f"Loaded {len(data)} cases from {gt_file}")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def match_data(self, groundtruth, metrics):
        """匹配 groundtruth 窗口与 metric 数据"""
        if groundtruth.empty or metrics.empty:
            logger.error("Empty groundtruth or metrics data")
            return []
        
        metric_time_range = pd.to_datetime(metrics['timestamp'], unit='ms')
        logger.info(f"Metric data range: {metric_time_range.min()} to {metric_time_range.max()}")
        
        matched_data = []
        for idx, gt_row in groundtruth.iterrows():
            window_start = gt_row['st_time']
            window_end = gt_row['ed_time']
            window_data = metrics[
                (metrics['timestamp'] >= pd.Timestamp(window_start).timestamp() * 1000) &
                (metrics['timestamp'] <= pd.Timestamp(window_end).timestamp() * 1000)
            ]
            if window_data.empty:
                logger.warning(f"No data in window: {window_start} to {window_end}")
            else:
                logger.info(f"Matched data count for {window_start} to {window_end}: {len(window_data)}")
            matched_data.append((idx, window_data))
        return matched_data

class MetricEventProcessor(DataLoader):
    def __init__(self, data_dir, cases, config=None, detector_config=None, adaptive_config=False, debug=False, detailed_debug=False):
        super().__init__(data_dir, debug)
        self.cases = cases
        self.detailed_debug = detailed_debug
        self.adaptive_config = adaptive_config
        self.config = config if config else {'minute': 60000, 'MIN_TEST_LENGTH': 2}
        self.metric_configs = {}
        
        if not adaptive_config:
            self.detector_config = detector_config if detector_config else {'k_s': {'k_thr': 1.0, 'std_thr': 0.005, 'win_size': 8}}
            self.detector = Ksigma(self.detector_config)
        else:
            self.detector = None
            self.detector_config = None
        
        self.res = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))
        self.stats = {
            'total_metrics': 0, 'successful_reads': 0, 'total_detections': 0,
            'cases_with_data': 0, 'cases_without_data': 0, 'detection_failures': 0
        }

    def detect_anomalies_for_services(self, metric_data, start_ts, end_ts, kpi_name, case_id, metric):
        """对每个服务进行异常检测"""
        anomalies = []
        
        if 'PodName' not in metric_data.columns:
            if self.detailed_debug:
                logger.info(f"No PodName column in data for metric {metric}")
            return anomalies
        
        if self.detailed_debug:
            logger.info(f"Processing {metric_data['PodName'].nunique()} services for case {case_id}, metric {kpi_name}")
            logger.info(f"Window data points: {len(metric_data)}")
            logger.info(f"Value stats: min={metric_data['value'].min():.4f}, max={metric_data['value'].max():.4f}, mean={metric_data['value'].mean():.4f}, std={metric_data['value'].std():.4f}")
        
        services_processed = 0
        services_with_enough_data = 0
        services_with_anomalies = 0
        
        if self.adaptive_config and metric in self.metric_configs:
            detector = Ksigma(self.metric_configs[metric])
        else:
            detector = self.detector
        
        for cmdb_id in metric_data['PodName'].unique():
            services_processed += 1
            service_data = metric_data[metric_data['PodName'] == cmdb_id].copy()
            
            if len(service_data) < self.config['MIN_TEST_LENGTH']:
                if self.detailed_debug:
                    logger.info(f"Service {cmdb_id}: insufficient data ({len(service_data)} points)")
                continue
            
            services_with_enough_data += 1
            
            try:
                if 'value' in service_data.columns:
                    values = service_data['value']
                    value_mean = values.mean()
                    value_std = values.std()
                    value_cv = value_std / value_mean if value_mean != 0 else 0
                    if self.detailed_debug:
                        logger.info(f"Service {cmdb_id}: {len(service_data)} points, mean={value_mean:.4f}, std={value_std:.4f}, cv={value_cv:.4f}")
                
                res = detector.detection(service_data, 'value', start_ts / 1000, end_ts / 1000)
                if self.detailed_debug:
                    logger.info(f"Service {cmdb_id}: Score={res[2]:.4f}, Is_Anomaly={res[0]}, Anomaly_TS={res[1]}")
                if res[0] is True and res[2] > 0.1:  # 降低分数阈值
                    services_with_anomalies += 1
                    anomaly_ts_ms = int(res[1] * 1000) if res[1] else None
                    anomalies.append([anomaly_ts_ms, cmdb_id, kpi_name, res[2]])
                    if self.debug:
                        logger.info(f"ANOMALY! Service: {cmdb_id}, KPI: {kpi_name}, Score: {res[2]:.4f}")
            except Exception as e:
                self.stats['detection_failures'] += 1
                if self.debug:
                    logger.error(f"Error detecting anomaly for service {cmdb_id}: {e}")
                continue
        
        if self.debug:
            logger.info(f"Services summary: {services_processed} total, {services_with_enough_data} with enough data, {services_with_anomalies} with anomalies")
        
        return anomalies

    def process_data(self, save_path=None):
        """主流程：加载数据、匹配时间窗口、检测异常并保存结果"""
        metric_files = self.get_tt_metric_names()
        groundtruth = self.load_groundtruth()
        
        if groundtruth.empty:
            logger.error("No groundtruth data loaded")
            return
        
        self.stats['total_metrics'] = sum(len(files) for files in metric_files.values())
        
        for day, files in metric_files.items():
            for metric in files:
                parts = metric.split('_', 1)
                if len(parts) != 2:
                    logger.warning(f"Invalid metric name format: {metric}")
                    continue
                date, filename = parts
                metric_path = os.path.join(self.data_dir, date, 'metric', filename)
                metrics = self.read(metric_path, metric)
                
                if metrics.empty:
                    logger.warning(f"Skipping metric {metric}: No valid data loaded")
                    continue
                
                self.stats['successful_reads'] += 1
                
                # 自适应配置
                if self.adaptive_config:
                    self.metric_configs[metric] = compute_adaptive_config(metrics)
                
                # 提取 KPI 名称
                kpi_name = filename.replace('.csv', '')
                
                # 过滤对应日期的 groundtruth
                day_groundtruth = groundtruth[groundtruth['st_time'].dt.strftime('%Y-%m-%d') == day]
                if day_groundtruth.empty:
                    logger.warning(f"No groundtruth data for {day}")
                    continue
                
                # 匹配时间窗口
                matched_data = self.match_data(day_groundtruth, metrics)
                
                # 对每个时间窗口进行异常检测
                for case_id, window_data in matched_data:
                    if window_data.empty:
                        self.stats['cases_without_data'] += 1
                        if self.detailed_debug:
                            logger.info(f"Case {case_id}, Metric {kpi_name}: No data in window")
                        continue
                    
                    self.stats['cases_with_data'] += 1
                    self.stats['total_detections'] += 1
                    
                    start_ts = time_to_ts(groundtruth.loc[case_id, 'st_time']) - self.config['minute'] * 5  # 缩小扩展范围
                    end_ts = time_to_ts(groundtruth.loc[case_id, 'ed_time']) + self.config['minute'] * 2
                    
                    if self.detailed_debug:
                        logger.info(f"Case {case_id}, Metric {kpi_name}:")
                        logger.info(f"Window: {ts_to_date(start_ts / 1000)} to {ts_to_date(end_ts / 1000)}")
                        logger.info(f"Data range: {ts_to_date(metrics['timestamp'].min() / 1000)} to {ts_to_date(metrics['timestamp'].max() / 1000)}")
                        logger.info(f"Data points: {len(window_data)}")
                        if 'PodName' in window_data.columns:
                            logger.info(f"Services: {window_data['PodName'].nunique()}, sample: {list(window_data['PodName'].unique()[:3])}")
                    
                    anomalies = self.detect_anomalies_for_services(
                        window_data, start_ts, end_ts, kpi_name, case_id, metric
                    )
                    for anomaly in anomalies:
                        self.res[case_id].append(anomaly)
        
        self.print_stats()
        
        if save_path:
            self.save_res(save_path)

    def print_stats(self):
        """打印统计信息"""
        logger.info(f"\n=== Detection Statistics ===")
        logger.info(f"Total metrics: {self.stats['total_metrics']}")
        logger.info(f"Successfully read: {self.stats['successful_reads']}")
        logger.info(f"Total detections attempted: {self.stats['total_detections']}")
        logger.info(f"Cases with data: {self.stats['cases_with_data']}")
        logger.info(f"Cases without data: {self.stats['cases_without_data']}")
        logger.info(f"Detection failures: {self.stats['detection_failures']}")
        
        total_anomalies = sum(len(events) for events in self.res.values())
        cases_with_anomalies = sum(1 for events in self.res.values() if len(events) > 0)
        
        logger.info(f"Total anomalies detected: {total_anomalies}")
        logger.info(f"Cases with anomalies: {cases_with_anomalies}/{len(self.cases)}")

    def save_res(self, save_path):
        """保存结果到 JSON 文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.res, f, indent=2)
        logger.info(f'Results saved successfully to: {save_path}')

def run_detection_with_config(data_dir, config_name, detector_config=None, adaptive_config=False, save_path_base=None):
    """运行检测流程"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running detection with {config_name} configuration")
    if adaptive_config:
        logger.info("Using adaptive configuration per metric")
    else:
        logger.info(f"Config: {detector_config}")
    logger.info(f"{'='*60}")
    
    # 加载 groundtruth
    loader = DataLoader(data_dir, debug=True)
    labels = loader.load_groundtruth()
    if labels.empty:
        logger.error("No groundtruth files found!")
        return
    
    # 初始化处理器
    processor = MetricEventProcessor(
        data_dir, labels, detector_config=detector_config,
        adaptive_config=adaptive_config, debug=True, detailed_debug=True
    )
    
    # 处理数据并保存结果
    save_path = save_path_base if save_path_base else f'/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/liningshuai/metric_events_tt_{config_name}.json'
    processor.process_data(save_path)
    
    # 打印结果摘要
    total_events = sum(len(events) for events in processor.res.values())
    cases_with_events = sum(1 for events in processor.res.values() if len(events) > 0)
    
    logger.info(f"\n=== {config_name} Results Summary ===")
    logger.info(f"Total fault cases: {len(labels)}")
    logger.info(f"Cases with detected events: {cases_with_events}")
    logger.info(f"Total anomalous metric events: {total_events}")
    logger.info(f"Results saved to: {save_path}")
    
    logger.info(f"\n=== {config_name} Case-by-Case Results ===")
    for case_id, events in processor.res.items():
        if len(events) > 0:
            logger.info(f"Case {case_id}: {len(events)} events")
            for event in events[:3]:
                if len(event) >= 4:
                    timestamp, cmdb_id, kpi_name, score = event
                    readable_time = ts_to_date(timestamp / 1000) if timestamp else "Unknown"
                    logger.info(f"  - {readable_time}: {cmdb_id} - {kpi_name} (score: {score:.3f})")
                if len(events) > 3:
                    logger.info(f"  ... and {len(events) - 3} more events")
        else:
            logger.info(f"Case {case_id}: No events detected")

if __name__ == "__main__":
    data_dir = "/home/fuxian/DataSet/tt"
    
    configs = {
        'adaptive': None,
        'default': {
            'k_s': {
                'k_thr': 1.0,   # 调整为更合理的阈值
                'std_thr': 0.005,  # 放宽标准差阈值
                'win_size': 8   # 适配 60s 采样
            }
        }
    }
    
    for config_name, detector_config in configs.items():
        run_detection_with_config(
            data_dir, config_name, detector_config,
            adaptive_config=(config_name == 'adaptive')
        )
