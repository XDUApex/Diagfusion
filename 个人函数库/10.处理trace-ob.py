# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')))
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import json
from detector.k_sigma import Ksigma
import public_function as pf
from tqdm import tqdm

tz = pytz.timezone('Asia/Shanghai')

def ts_to_date(timestamp):
    try:
        return datetime.fromtimestamp(timestamp // 1000, tz).strftime('%Y-%m-%d %H:%M:%S.%f')
    except:
        return datetime.fromtimestamp(timestamp // 1000, tz).strftime('%Y-%m-%d %H:%M:%S')

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

class TraceUtils:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pairs = self.getPairs()
        
    def getPairs(self):
        pairs = set()
        tt_days = ['29', '30']  # 2023-01-29, 2023-01-30
        for day_str in tt_days:
            filepath = os.path.join(self.data_dir, f'2023-01-{day_str}', 'trace', 'trace.csv')
            if not os.path.exists(filepath):
                print(f"Trace file not found: {filepath}")
                continue
            df = pd.read_csv(filepath)
            # Extract service call pairs through parent_id and span_id matching
            cdata = df[['parent_id', 'service_name']].rename(columns={'parent_id': 'span_id', 'service_name': 'cservice_name'})
            merged = pd.merge(df, cdata, on='span_id', how='left')
            for _, row in merged.iterrows():
                caller = row['service_name']
                callee = row['cservice_name']
                if pd.notna(caller) and pd.notna(callee) and caller != callee:
                    pairs.add((caller, callee))
        
        print(f"Discovered {len(pairs)} service pairs: {list(pairs)[:10]}...")
        return list(pairs)
    
    def get_trace_by_day(self, date_str: str):
        filepath = os.path.join(self.data_dir, f'2023-01-{date_str}', 'trace', 'trace.csv')
        if not os.path.exists(filepath):
            print(f"Trace file not found: {filepath}")
            return pd.DataFrame()
        df = pd.read_csv(filepath)
        # Convert timestamps to milliseconds and adjust from UTC+8 to UTC
        df['timestamp'] = (df['timestamp'] - 28800) * 1000  # Subtract 8 hours (28800 seconds)
        df['st_time'] = (df['st_time'] - 28800) * 1000
        df['ed_time'] = (df['ed_time'] - 28800) * 1000
        df['lagency'] = df['ed_time'] - df['st_time']  # Calculate latency
        return df

    def data_process(self, date_str: str):
        data = self.get_trace_by_day(date_str)
        if data.empty:
            return pd.DataFrame()
        cdata = data[['parent_id', 'service_name']].rename(columns={'parent_id': 'span_id', 'service_name': 'cservice_name'})
        return pd.merge(data, cdata, on='span_id')
    
    def infer_status_from_message(self, message):
        """
        Infer call status from message field
        """
        if pd.isna(message):
            return 200  # Default to success
        
        message = str(message).lower()
        
        # Error keywords
        error_keywords = [
            'error', 'failed', 'fail', 'exception', 'timeout', 
            'panic', 'abort', 'crash', 'denied', 'refused',
            'unavailable', 'deadline', 'cancelled', 'invalid',
            'not found', '404', '500', '503', 'internal server error',
            'connection refused', 'connection timeout', 'network error'
        ]
        
        # Check if contains error keywords
        for keyword in error_keywords:
            if keyword in message:
                return 500
        
        # Default to success
        return 200
    
    def turn_to_timeseries(self, date_str, savepath):
        day_path = os.path.join(savepath, f'2023-01-{date_str}')
        if not os.path.exists(day_path):
            os.makedirs(day_path)
        df = self.data_process(date_str)
        if df.empty:
            print(f"No data for day 2023-01-{date_str}")
            return
        df['timestamp'] = df['timestamp'].apply(lambda x: int(x))  # Ensure timestamp is integer
        
        # Infer status based on message field
        print(f"No status_code column found for day {date_str}, inferring status from message field")
        df['mapped_status'] = df['message'].apply(self.infer_status_from_message)
        
        # Statistics of status distribution
        status_counts = df['mapped_status'].value_counts()
        print(f"Status distribution for day {date_str}: {dict(status_counts)}")
        
        date = ts_to_date(df['timestamp'].iloc[0]).split()[0] if not df.empty else f'2023-01-{date_str}'
        start_ts = time_to_ts(date)
        delta = 30000  # 30s per point
        points_count = 60000 * 24 * 60 // delta
        ts = [start_ts + delta * i for i in range(1, points_count + 1)]
        
        processed_pairs = 0
        for caller, callee in self.pairs:
            temp = df.loc[(df['service_name'] == caller) & (df['cservice_name'] == callee)]
            if temp.empty:
                continue  # Skip service pairs with no data
                
            print(f"Processing {date_str}: {caller} -> {callee} ({len(temp)} records)")
            info = {'timestamp': ts, '200': [], '500': [], 'other': [], 'lagency': []}
            
            for k in range(points_count):
                chosen = temp.loc[(temp['timestamp'] >= start_ts + k * delta) &
                                  (temp['timestamp'] < start_ts + (k + 1) * delta)]
                cur_lagency = max(0, np.mean(chosen['lagency'].values)) if not chosen.empty else 0
                
                # Statistics based on inferred status
                cur_200 = len(chosen.loc[chosen['mapped_status'] == 200]) if not chosen.empty else 0
                cur_500 = len(chosen.loc[chosen['mapped_status'] == 500]) if not chosen.empty else 0
                cur_other = len(chosen) - cur_200 - cur_500 if not chosen.empty else 0
                
                info['lagency'].append(cur_lagency)
                info['200'].append(cur_200)
                info['500'].append(cur_500)
                info['other'].append(cur_other)
            
            output_file = os.path.join(day_path, f'{caller}_{callee}.csv')
            pd.DataFrame(info).to_csv(output_file, index=False)
            processed_pairs += 1
        
        print(f"Generated {processed_pairs} CSV files for {date_str}")

class InvocationEvent:
    def __init__(self, cases, data_dir, dataset, sequence_path=None, config=None):
        self.cases = cases
        self.data_dir = data_dir
        self.dataset = dataset
        self.sequence_path = sequence_path
        self.detector = Ksigma()
        self.pairs = self.getPairs()
        if config is None:
            config = {}
            config['minute'] = 60000
            config['MIN_TEST_LENGTH'] = 5
        self.config = config
        self.res_original = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))
        self.res_offset = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))
        
    def getPairs(self):
        if self.dataset == 'gaia':
            services_pairs = {
                "checkoutservice": [
                    "paymentservice",
                    "shippingservice",
                    "cartservice",
                    "productcatalogservice",
                    "emailservice"
                ],
                "frontend": [
                    "cartservice",
                    "recommendationservice",
                    "checkoutservice",
                    "currencyservice",
                    "adservice",
                    "productcatalogservice",
                    "shippingservice"
                ],
                "recommendationservice": [
                    "productcatalogservice"
                ]
            }
            pairs = []
            for caller in services_pairs:
                for callee in services_pairs[caller]:
                    for i in [1, 2]:
                        for j in [1, 2]:
                            pairs.append((caller + str(i), callee + str(j)))
            pairs.extend([('logservice1', 'logservice2'), ('logservice2', 'logservice1')])
        elif self.dataset == 'tt':  # TT dataset
            # Discover service pairs from processed data directory
            pairs = set()
            if os.path.exists(self.data_dir):
                # Traverse all date directories
                for date_dir in os.listdir(self.data_dir):
                    if date_dir.startswith('2023-01-'):  # TT dataset date prefix
                        date_path = os.path.join(self.data_dir, date_dir)
                        if os.path.isdir(date_path):
                            # Extract service pairs from CSV filenames
                            for csv_file in os.listdir(date_path):
                                if csv_file.endswith('.csv') and '_' in csv_file:
                                    # Parse caller_callee.csv format
                                    parts = csv_file[:-4].split('_', 1)  # Remove .csv, split once
                                    if len(parts) == 2:
                                        caller, callee = parts
                                        pairs.add((caller, callee))
            
            if pairs:
                print(f"Found {len(pairs)} service pairs from processed data")
                return list(pairs)
            else:
                # If no processed data found, generate from raw data
                print("No processed data found, generating from raw data...")
                tt_days = ['29', '30']  # TT dataset dates
                # Get original data directory
                original_data_dir = '/home/fuxian/DataSet/NewDataset/tt'  # TT dataset path
                for day_str in tt_days:
                    filepath = os.path.join(original_data_dir, f'2023-01-{day_str}', 'trace', 'trace.csv')
                    if not os.path.exists(filepath):
                        continue
                    df = pd.read_csv(filepath)
                    cdata = df[['parent_id', 'service_name']].rename(columns={'parent_id': 'span_id', 'service_name': 'cservice_name'})
                    merged = pd.merge(df, cdata, on='span_id', how='left')
                    for _, row in merged.iterrows():
                        caller = row['service_name']
                        callee = row['cservice_name']
                        if pd.notna(caller) and pd.notna(callee) and caller != callee:
                            pairs.add((caller, callee))
                return list(pairs)
        else:
            raise Exception(f"Unsupported dataset: {self.dataset}")
        return pairs
    
    def read(self, day, caller, callee):
        filepath = os.path.join(self.data_dir, day, f'{caller}_{callee}.csv')
        if not os.path.exists(filepath):
            return pd.DataFrame()
        data = pd.read_csv(filepath)
        if 'timestamp' not in data.columns:
            print(f"Warning: timestamp column not found in {filepath}")
            return pd.DataFrame()
        data.index = [ts_to_date(ts) for ts in data['timestamp']]
        return data
    
    def get_invocation_events(self):
        print(f"Processing {len(self.cases)} fault cases with {len(self.pairs)} service pairs")
        
        for case_id, case in tqdm(self.cases.iterrows(), desc="Processing cases"):
            day = case['date']  # Use date column from groundtruth.csv
            case_events_original = 0
            case_events_offset = 0
            
            for caller, callee in self.pairs:
                invocation_data = self.read(day, caller, callee)
                if invocation_data.empty:
                    continue
                
                # Original timestamps
                start_ts_original = time_to_ts(case['st_time']) - self.config['minute'] * 31
                end_ts_original = time_to_ts(case['ed_time']) + self.config['minute'] * 1
                
                # 8-hour offset timestamps
                start_ts_offset = time_to_ts(case['st_time'], add_8_hours=True) - self.config['minute'] * 31
                end_ts_offset = time_to_ts(case['ed_time'], add_8_hours=True) + self.config['minute'] * 1
                
                # Detect latency anomalies (original)
                res1_original = self.detector.detection(invocation_data, 'lagency', start_ts_original, end_ts_original)
                
                # Detect 500 status code anomalies (original)
                res2_original = (False, None, 0)
                if '500' in invocation_data.columns and invocation_data['500'].sum() > 0:
                    res2_original = self.detector.detection(invocation_data, '500', start_ts_original, end_ts_original)
                
                # Detect latency anomalies (offset)
                res1_offset = self.detector.detection(invocation_data, 'lagency', start_ts_offset, end_ts_offset)
                
                # Detect 500 status code anomalies (offset)
                res2_offset = (False, None, 0)
                if '500' in invocation_data.columns and invocation_data['500'].sum() > 0:
                    res2_offset = self.detector.detection(invocation_data, '500', start_ts_offset, end_ts_offset)
                
                # Process original timestamps
                if res1_original[0] or res2_original[0]:
                    ts = None
                    score = 0
                    anomaly_type = "latency"
                    
                    if res1_original[0]:
                        ts = res1_original[1]
                        score = res1_original[2]
                        anomaly_type = "latency"
                    
                    if res2_original[0]:
                        if ts is None:
                            ts = res2_original[1]
                            score = res2_original[2]
                            anomaly_type = "error_rate"
                        else:
                            if res2_original[1] < ts:
                                ts = res2_original[1]
                                score = res2_original[2]
                                anomaly_type = "error_rate"
                    
                    self.res_original[case_id].append((int(ts), caller, callee, score, anomaly_type))
                    case_events_original += 1
                
                # Process offset timestamps
                if res1_offset[0] or res2_offset[0]:
                    ts = None
                    score = 0
                    anomaly_type = "latency"
                    
                    if res1_offset[0]:
                        ts = res1_offset[1]
                        score = res1_offset[2]
                        anomaly_type = "latency"
                    
                    if res2_offset[0]:
                        if ts is None:
                            ts = res2_offset[1]
                            score = res2_offset[2]
                            anomaly_type = "error_rate"
                        else:
                            if res2_offset[1] < ts:
                                ts = res2_offset[1]
                                score = res2_offset[2]
                                anomaly_type = "error_rate"
                    
                    self.res_offset[case_id].append((int(ts), caller, callee, score, anomaly_type))
                    case_events_offset += 1
            
            if case_events_original > 0:
                print(f"Case {case_id} (original): Found {case_events_original} anomalous invocation events")
            if case_events_offset > 0:
                print(f"Case {case_id} (offset): Found {case_events_offset} anomalous invocation events")
                
    def save_res(self, savepath_original, savepath_offset):
        os.makedirs(os.path.dirname(savepath_original), exist_ok=True)
        os.makedirs(os.path.dirname(savepath_offset), exist_ok=True)
        with open(savepath_original, 'w') as f:
            json.dump(self.res_original, f, indent=2)
        print(f'Original results saved successfully to: {savepath_original}')
        with open(savepath_offset, 'w') as f:
            json.dump(self.res_offset, f, indent=2)
        print(f'Offset results saved successfully to: {savepath_offset}')

if __name__ == '__main__':
    # Load configuration file
    try:
        config = pf.get_config()
    except:
        # If configuration file has issues, use default configuration
        config = {'dataset': 'tt'}  # TT dataset
    
    project_root_dir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..'))
    
    # Define TT dataset dates
    tt_days = ['29', '30']  # 2023-01-29, 2023-01-30
    
    # Read groundtruth.csv files for all dates
    labels = []
    data_base_dir = '/home/fuxian/DataSet/NewDataset/tt'  # TT dataset path
    
    print("=== Loading groundtruth files ===")
    for day_str in tt_days:
        label_path = os.path.join(data_base_dir, f'2023-01-{day_str}', 'groundtruth.csv')
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            print(f"Loaded {len(df)} cases from {label_path}")
            labels.append(df)
        else:
            print(f"Missing: {label_path}")
    
    if not labels:
        print("No groundtruth files found!")
        sys.exit(1)
        
    labels = pd.concat(labels, ignore_index=True)
    print(f"Total loaded: {len(labels)} fault cases")
    
    # Set save paths
    save_path_original = os.path.abspath('/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/trace_events_tt_original.json')
    save_path_offset = os.path.abspath('/home/fuxian/lixuyang/fuxian/DiagFusion25+tt/个人函数库/trace_events_tt_offset_8h.json')
    
    # Preprocess trace data, generate time series
    print("\n=== Processing trace data to time series ===")
    trace_utils = TraceUtils(data_base_dir)
    trace_temp_dir = os.path.join(project_root_dir, 'trace_temp_tt')  # TT dataset temp directory
    
    for day_str in tt_days:
        print(f"\nProcessing day 2023-01-{day_str}")
        trace_utils.turn_to_timeseries(day_str, trace_temp_dir)
    
    # Anomaly detection
    print("\n=== Starting invocation event detection ===")
    invocation_event = InvocationEvent(labels, trace_temp_dir, 'tt')  # TT dataset
    print(f"Service pairs to analyze: {len(invocation_event.pairs)}")
    
    invocation_event.get_invocation_events()
    invocation_event.save_res(save_path_original, save_path_offset)
    
    # Statistics summary
    for label, res in [("Original", invocation_event.res_original), ("Offset 8h", invocation_event.res_offset)]:
        total_events = sum(len(events) for events in res.values())
        cases_with_events = sum(1 for events in res.values() if len(events) > 0)
        latency_events = sum(sum(1 for event in events if len(event) > 4 and event[4] == "latency") for events in res.values())
        error_events = sum(sum(1 for event in events if len(event) > 4 and event[4] == "error_rate") for events in res.values())
        
        print(f"\n=== {label} Results Summary ===")
        print(f"Total fault cases: {len(labels)}")
        print(f"Cases with detected events: {cases_with_events}")
        print(f"Total anomalous invocation events: {total_events}")
        print(f"  - Latency anomalies: {latency_events}")
        print(f"  - Error rate anomalies: {error_events}")
        print(f"Results saved to: {save_path_original if label == 'Original' else save_path_offset}")
        
        # Show some sample results
        print(f"\n=== {label} Sample Results ===")
        sample_count = 0
        for case_id, events in res.items():
            if events and sample_count < 3:  # Show first 3 cases with results
                print(f"Case {case_id}: {len(events)} events")
                for event in events[:2]:  # Show first 2 events for each case
                    if len(event) >= 5:
                        timestamp, caller, callee, score, anomaly_type = event
                        readable_time = ts_to_date(timestamp)
                        print(f"  - {readable_time}: {caller} -> {callee} ({anomaly_type}, score: {score:.3f})")
                    else:
                        timestamp, caller, callee, score = event
                        readable_time = ts_to_date(timestamp)
                        print(f"  - {readable_time}: {caller} -> {callee} (score: {score:.3f})")
                if len(events) > 2:
                    print(f"  ... and {len(events) - 2} more events")
                sample_count += 1