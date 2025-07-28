import datetime

def analyze_timestamps_from_data():
    """
    分析给定数据中的时间戳范围
    """
    # 您提供的数据
    data = [
        [1625330250000, "webservice1", "mobservice1", 3.5687984077259056], 
        [1625330250000, "webservice2", "mobservice1", 2.3067597550651238], 
        [1625330250000, "webservice1", "redisservice1", 1.5735517937451509], 
        [1625330250000, "webservice2", "redisservice1", 1.6318629171251797], 
        [1625330250000, "mobservice1", "redisservice1", 1.7950549357115015], 
        [1625330250000, "mobservice1", "redisservice2", 1.7950549357115015], 
        [1625330250000, "logservice1", "dbservice1", 1.2848617668395412], 
        [1625330250000, "logservice2", "dbservice1", 2.1206906483279915], 
        [1625330190000, "logservice2", "dbservice2", 2.7970963453121107], 
        [1625330190000, "logservice1", "redisservice2", 1.025589904412864], 
        [1625330250000, "logservice2", "redisservice1", 1.9034803248929624], 
        [1625330250000, "dbservice1", "redisservice1", 2.9401283223287416], 
        [1625330280000, "dbservice2", "redisservice1", 1.5601833944063435], 
        [1625330190000, "dbservice2", "redisservice2", 2.1603278390868383]
    ]
    
    # 提取所有时间戳
    timestamps = [entry[0] for entry in data]
    
    # 找出最大最小值
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    
    # 转换为可读格式
    def timestamp_to_datetime(timestamp):
        if len(str(int(timestamp))) > 10:
            timestamp_sec = timestamp / 1000
        else:
            timestamp_sec = timestamp
        
        dt = datetime.datetime.fromtimestamp(timestamp_sec)
        return dt.strftime("%Y年%m月%d日 %H时%M分%S秒")
    
    print("=== 时间戳分析结果 ===")
    print(f"数据总条数: {len(data)}")
    print(f"唯一时间戳数量: {len(set(timestamps))}")
    print()
    
    print("时间戳统计:")
    timestamp_counts = {}
    for ts in timestamps:
        timestamp_counts[ts] = timestamp_counts.get(ts, 0) + 1
    
    for ts, count in sorted(timestamp_counts.items()):
        print(f"  {ts} -> {timestamp_to_datetime(ts)} (出现{count}次)")
    
    print()
    print("时间范围:")
    print(f"最早时间戳: {min_timestamp}")
    print(f"最早时间: {timestamp_to_datetime(min_timestamp)}")
    print()
    print(f"最晚时间戳: {max_timestamp}")
    print(f"最晚时间: {timestamp_to_datetime(max_timestamp)}")
    print()
    print(f"时间跨度: {max_timestamp - min_timestamp} 毫秒")
    print(f"时间跨度: {(max_timestamp - min_timestamp) / 1000} 秒")
    print(f"时间跨度: {(max_timestamp - min_timestamp) / 1000 / 60} 分钟")

if __name__ == "__main__":
    analyze_timestamps_from_data()