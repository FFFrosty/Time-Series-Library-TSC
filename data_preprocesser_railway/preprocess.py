import numpy as np
import pandas as pd
import os
import glob


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features"""
    df = df.copy()

    # Basic numerical features/基础流量
    numeric_features = ['inbound_count', 'outbound_count', 'total_flow', 'net_flow']

    # Time features/时间特征
    time_features = ['hour', 'minute', 'weekday']

    # Convert boolean features to numerical/布尔特征
    bool_features = ['is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour']
    for col in bool_features:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Create periodic features/周期性特征，避免边界断点
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Lag features/滞后特征
    for lag in [1, 2, 3, 6, 12, 144]:
        df[f'total_flow_lag_{lag}'] = df['total_flow'].shift(lag)
        df[f'inbound_lag_{lag}'] = df['inbound_count'].shift(lag)
        df[f'outbound_lag_{lag}'] = df['outbound_count'].shift(lag)

    # Rolling window statistical features/滑动窗口
    for window in [6, 12, 24]:
        df[f'total_flow_ma_{window}'] = df['total_flow'].rolling(window=window).mean()
        df[f'total_flow_std_{window}'] = df['total_flow'].rolling(window=window).std()

    # Remove rows containing NaN
    df = df.dropna().reset_index(drop=True)

    return df


# ================= 数据聚合与输出部分 =================

DATA_DIR = "data_preprocesser_railway/processed_data_by_station"
PATTERN = "station_*.csv"
SINGLE_STATION_FILE = None
NUM_STATIONS = 55
# 注意：如果你只要 total_flow，其实不需要跑上面的 create_features
TARGET_FEATURE = 'total_flow'


def list_station_files():
    if SINGLE_STATION_FILE:
        return [os.path.join(DATA_DIR, SINGLE_STATION_FILE)]
    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN)))
    if PATTERN.endswith("*.csv"):
        files = files[:NUM_STATIONS]
    return files


def main():
    files = list_station_files()
    if not files:
        raise FileNotFoundError("No station files found. Check DATA_DIR / PATTERN / SINGLE_STATION_FILE.")

    print(f"Total stations found: {len(files)}")
    print(f"Example file: {os.path.basename(files[0])}")

    df_list = []

    for fp in files:
        # 1. 解析文件名: 'station_lineID_stationID.csv' -> 'stationID_lineID'
        base_name = os.path.basename(fp).replace(".csv", "")
        parts = base_name.split('_')
        if len(parts) >= 3:
            line_id = parts[1]
            station_id = parts[2]
            col_name = f"{station_id}_{line_id}"
        else:
            col_name = base_name

            # 2. 读取与特征处理
        df = pd.read_csv(fp)
        df_feat = df

        if 'datetime_slot' not in df_feat.columns:
            continue

        # 3. 提取目标列
        df_subset = df_feat[['datetime_slot', TARGET_FEATURE]].copy()

        # ✨ 关键修复 1：在设定索引前，就将时间统一格式化为 datetime 对象
        df_subset['datetime_slot'] = pd.to_datetime(df_subset['datetime_slot'], format='mixed')

        df_subset.set_index('datetime_slot', inplace=True)

        # ✨ 关键修复 2：以防单个站点本身就有重复的时间戳，保留第一条即可
        df_subset = df_subset[~df_subset.index.duplicated(keep='first')]

        # 4. 重命名并加入列表
        df_subset.rename(columns={TARGET_FEATURE: col_name}, inplace=True)
        df_list.append(df_subset)

    print("Merging data...")
    # 5. 此时 concat 依据的是标准的时间戳，不再会有因为字符串格式差异导致的错位
    final_df = pd.concat(df_list, axis=1)

    # 6. 整理 DataFrame
    final_df.index.name = 'date'
    final_df.reset_index(inplace=True)

    # 因为索引已经是 datetime 类型，这里不需要再次转换了，直接排序即可
    final_df.sort_values('date', inplace=True)
    final_df.fillna(0, inplace=True)

    # 7. 保存输出
    output_filename = "railway_passenger_flow.csv"
    final_df.to_csv(output_filename, index=False)

    print(f"Successfully saved to {output_filename}")
    print(f"Final output shape: {final_df.shape}")
    print(final_df.head())


if __name__ == "__main__":
    main()