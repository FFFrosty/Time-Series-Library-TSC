import numpy as np
import pandas as pd
import os
import glob

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features"""
    df = df.copy()

    # Basic numerical features/基础流量
    numeric_features = ['inbound_count', 'outbound_count', 'total_flow', 'net_flow']
    # net_flow=inbound_count-outbound_count,净客流
    # total_flow=inbound_count+outbound_count，总客流
    # Time features/时间特征
    time_features = ['hour', 'minute', 'weekday']

    # Convert boolean features to numerical/布尔特征
    bool_features = ['is_weekend', 'is_morning_rush', 'is_evening_rush',
                     'is_rush_hour']  # 'is_total_flow_outlier', 'is_inbound_outlier', 'is_outbound_outlier']
    # 原始数据里没有outlier类的数据
    for col in bool_features:
        df[col] = df[col].astype(int)

    # Create periodic features/周期性特征，避免边界断点
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Lag features/滞后特征，对总客流、进站、出站做历史值，当前时刻往前第n个时间片的值
    for lag in [1, 2, 3, 6, 12, 144]:  # 10min, 20min, 30min, 1hour, 2hours, 1day
        # 数据集是1,2,3,6,12,24
        df[f'total_flow_lag_{lag}'] = df['total_flow'].shift(lag)
        df[f'inbound_lag_{lag}'] = df['inbound_count'].shift(lag)
        df[f'outbound_lag_{lag}'] = df['outbound_count'].shift(lag)

    # Rolling window statistical features/滑动窗口的均值和标准差，实际数据集里似乎还有最大值、最小值
    for window in [6, 12, 24]:  # 1hour, 2hours, 4hours
        df[f'total_flow_ma_{window}'] = df['total_flow'].rolling(window=window).mean()
        df[f'total_flow_std_{window}'] = df['total_flow'].rolling(window=window).std()

    # Remove rows containing NaN
    df = df.dropna().reset_index(drop=True)

    return df


DATA_DIR = "processed_data_by_station"
PATTERN = "station_*.csv"      # 多站点：station_*.csv；单站点：例如 station_A_67.csv
SINGLE_STATION_FILE = None# 例如 "station_A_67.csv"；若填写则忽略 PATTERN/NUM_STATIONS
NUM_STATIONS = 55

def list_station_files():
    if SINGLE_STATION_FILE:
        return [os.path.join(DATA_DIR, SINGLE_STATION_FILE)]
    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN)))
    if PATTERN.endswith("*.csv"):
        files = files[:NUM_STATIONS]
    return files


files = list_station_files()
if not files:
    raise FileNotFoundError("No station files found. Check DATA_DIR / PATTERN / SINGLE_STATION_FILE.")

print("Stations:", len(files))
print("Example:", os.path.basename(files[0]))

FEATURE_COLUMNS = [
    'inbound_count', 'outbound_count', 'total_flow', 'net_flow',
    'hour', 'minute', 'weekday',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'minute_sin', 'minute_cos',
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
    'total_flow_lag_1', 'total_flow_lag_2', 'total_flow_lag_3',
    'total_flow_lag_6', 'total_flow_lag_12', 'total_flow_lag_144',
    'inbound_lag_1', 'inbound_lag_2', 'inbound_lag_3',
    'inbound_lag_6', 'inbound_lag_12', 'inbound_lag_144',
    'outbound_lag_1', 'outbound_lag_2', 'outbound_lag_3',
    'outbound_lag_6', 'outbound_lag_12', 'outbound_lag_144',
    'total_flow_ma_6', 'total_flow_ma_12', 'total_flow_ma_24',
    'total_flow_std_6', 'total_flow_std_12', 'total_flow_std_24'
]
for fp in files:
    name = os.path.basename(fp).replace(".csv","")
    df = pd.read_csv(fp)
    df_feat = create_features(df)
    df_feat = df_feat[FEATURE_COLUMNS].values

debug = 0
