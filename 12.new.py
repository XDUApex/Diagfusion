import os
import pandas as pd
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir, debug=False):
        self.data_dir = data_dir
        self.debug = debug
        self.debug_count = 0

    def get_tt_metric_names(self, data_dir):
        """获取指定日期的 metric 文件列表，按日期分组"""
        metric_files = {}
        tt_days = ['2023-01-29', '2023-01-30']
        for day in tt_days:
            metric_dir = os.path.join(data_dir, day, 'metric')
            if os.path.exists(metric_dir):
                metric_files[day] = [f for f in os.listdir(metric_dir) if f.endswith('.csv')]
                if self.debug and self.debug_count < 5:
                    logger.info(f"Found metric files for {day}: {metric_files[day]}")
                    self.debug_count += 1
        return metric_files

    def read(self, file_path):
        """读取 CSV 文件并处理时间戳"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        data = pd.read_csv(file_path)
        if 'TimeStamp' in data.columns and pd.api.types.is_numeric_dtype(data['TimeStamp']):
            timestamp_col = 'TimeStamp'
            data['timestamp'] = data[timestamp_col] * 1000  # 秒转毫秒
            if self.debug and self.debug_count < 5:
                logger.info(f"Raw TimeStamp (first 5) for {file_path}: {data[timestamp_col].head().tolist()}")
                logger.info(f"Converted timestamp (first 5): {data['timestamp'].head().tolist()}")
                self.debug_count += 1
        elif 'Time' in data.columns:
            data['timestamp'] = pd.to_datetime(data['Time']).astype('int64') // 10**6  # 毫秒
        else:
            logger.warning(f"No valid timestamp column found in {file_path}")
            data['timestamp'] = pd.to_datetime('1970-01-01').astype('int64')  # 默认值
        
        return data

    def load_groundtruth(self, data_dir):
        """加载 groundtruth 文件"""
        gt_files = [f for f in os.listdir(data_dir) if f.endswith('groundtruth.csv')]
        all_data = []
        for gt_file in gt_files:
            file_path = os.path.join(data_dir, gt_file)
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                # 确保 st_time 和 ed_time 是 datetime 类型
                data['st_time'] = pd.to_datetime(data['st_time'])
                data['ed_time'] = pd.to_datetime(data['ed_time'])
                all_data.append(data)
            else:
                logger.warning(f"Groundtruth file not found: {file_path}")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def match_data(self, groundtruth, metrics):
        """匹配 groundtruth 窗口与 metric 数据"""
        if groundtruth.empty or metrics.empty:
            logger.error("Empty groundtruth or metrics data")
            return []

        # 打印 metric 数据范围
        metric_time_range = pd.to_datetime(metrics['timestamp'], unit='ms')
        logger.info(f"Metric data range: {metric_time_range.min()} to {metric_time_range.max()}")

        matched_data = []
        for _, gt_row in groundtruth.iterrows():
            window_start = gt_row['st_time']
            window_end = gt_row['ed_time']
            window_data = metrics[
                (metrics['timestamp'] >= pd.Timestamp(window_start).timestamp() * 1000) &
                (metrics['timestamp'] <= pd.Timestamp(window_end).timestamp() * 1000)
            ]
            if len(window_data) == 0:
                logger.warning(f"No data in window: {window_start} to {window_end}")
            else:
                logger.info(f"Matched data count for {window_start} to {window_end}: {len(window_data)}")
            matched_data.append(window_data)
        return matched_data

    def process_data(self):
        """主流程：加载数据并匹配"""
        metric_files = self.get_tt_metric_names(self.data_dir)
        groundtruth = self.load_groundtruth(self.data_dir)

        if groundtruth.empty:
            logger.error("No groundtruth data loaded")
            return

        all_matched_data = []
        for day, files in metric_files.items():
            for metric_file in files:
                metric_path = os.path.join(self.data_dir, day, 'metric', metric_file)
                metrics = self.read(metric_path)
                if not metrics.empty:
                    # 过滤 groundtruth 数据，仅处理对应日期
                    day_groundtruth = groundtruth[groundtruth['st_time'].dt.strftime('%Y-%m-%d') == day]
                    if not day_groundtruth.empty:
                        matched = self.match_data(day_groundtruth, metrics)
                        all_matched_data.extend(matched)
                    else:
                        logger.warning(f"No groundtruth data for {day}")

        return all_matched_data

# 使用示例
if __name__ == "__main__":
    data_dir = "/home/fuxian/DataSet/tt"
    loader = DataLoader(data_dir, debug=True)
    matched_data = loader.process_data()

    if matched_data:
        for i, data in enumerate(matched_data):
            logger.info(f"Matched data {i} shape: {data.shape if not data.empty else 'Empty'}")
    else:
        logger.info("No matched data found")