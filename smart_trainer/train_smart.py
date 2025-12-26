
"""
train_smart.py: LightGBM Model Trainer for Mihomo Smart Node Selection (Final Fix v2)

修复记录：
1. [Critical] 修复参数名不匹配问题：同时支持 --output 和 --output-file，解决 CI 报错。
2. 自动识别数据量：如果真实测速数据不足，自动回退到模拟官方权重。
3. 移除重复 Log1p：防止数据被双重压缩。
4. 补全 RobustScaler 保存逻辑。
"""

import argparse
import logging
import os
import re
import sys
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler




GO_SOURCE_URL = "https://raw.githubusercontent.com/vernesong/mihomo/Alpha/component/smart/lightgbm/transform.go"
CACHE_DIR = Path("./cache")
GO_SOURCE_CACHE_PATH = CACHE_DIR / "transform.go.cache"


DEFAULT_MODEL_PATH = Path("../models/Model.bin")
DEFAULT_DATA_DIR = Path("../data")



BIASED_FEATURES = [
    'download_mb', 'upload_mb', 'traffic_density', 'traffic_ratio', 
    'last_used_seconds', 'duration_minutes', 
    'history_download_mb', 'history_upload_mb'
]


COMPLEX_FEATURES = [
    'asn_feature', 'country_feature', 'address_feature', 
    'port_feature', 'connection_type_feature'
]


CONTINUOUS_FEATURES = [
    'connect_time', 'latency', 
    'upload_mb', 'download_mb', 
    'maxuploadrate_kb', 'maxdownloadrate_kb',
    'history_upload_mb', 'history_download_mb',
    'history_maxuploadrate_kb', 'history_maxdownloadrate_kb',
    'duration_minutes', 'last_used_seconds',
    'traffic_density', 'traffic_ratio',
    'asn_hash', 'host_hash', 'ip_hash', 'geoip_hash'
]


COUNT_FEATURES = ['success', 'failure'] 


LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)



def fetch_go_source() -> str:
    """下载或读取缓存的 transform.go"""
    if GO_SOURCE_CACHE_PATH.exists():

        if (time.time() - GO_SOURCE_CACHE_PATH.stat().st_mtime) < 86400:
            logging.info("Using cached transform.go")
            return GO_SOURCE_CACHE_PATH.read_text(encoding='utf-8')

    logging.info(f"Fetching Go source from {GO_SOURCE_URL}")
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        response = requests.get(GO_SOURCE_URL, timeout=10)
        response.raise_for_status()
        content = response.text
        GO_SOURCE_CACHE_PATH.write_text(content, encoding='utf-8')
        logging.info("Successfully fetched and cached Go source.")
        return content
    except Exception as e:
        if GO_SOURCE_CACHE_PATH.exists():
            logging.warning(f"Download failed ({e}), utilizing stale cache.")
            return GO_SOURCE_CACHE_PATH.read_text(encoding='utf-8')
        raise RuntimeError(f"Could not fetch transform.go and no cache available: {e}")

def parse_feature_order(go_content: str) -> dict:
    """从 Go 源码中解析特征 ID 映射"""
    logging.info("Parsing feature order from Go source...")

    func_match = re.search(r'func getDefaultFeatureOrder\(\) map\[int\]string \{(.*?)\}', go_content, re.DOTALL)
    if not func_match:
        logging.warning("Regex failed to find getDefaultFeatureOrder. Using fallback list.")
        return get_fallback_features()

    feature_map = {}

    pairs = re.findall(r'(\d+):\s*"([^"]+)"', func_match.group(1))
    for idx, name in pairs:
        feature_map[int(idx)] = name
    
    if not feature_map:
        return get_fallback_features()
    
    logging.info(f"Successfully parsed {len(feature_map)} features.")
    return feature_map

def get_fallback_features() -> dict:
    """如果解析失败的备用特征表"""
    features = [
        'success', 'failure', 'connect_time', 'latency', 'upload_mb', 'download_mb', 
        'duration_minutes', 'last_used_seconds', 'is_udp', 'is_tcp', 'asn_feature', 
        'country_feature', 'address_feature', 'port_feature', 'traffic_ratio', 
        'traffic_density', 'connection_type_feature', 'asn_hash', 'host_hash', 
        'ip_hash', 'geoip_hash'
    ]
    return {i: f for i, f in enumerate(features)}

def load_data(data_dir: Path, days: int = 15) -> pd.DataFrame:
    """加载最近 N 天的 CSV 数据"""
    logging.info(f"Loading data from the last {days} days from '{data_dir}'...")
    

    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = glob.glob(str(data_dir / "*.csv"))
    
    cutoff_time = time.time() - (days * 86400)
    recent_files = [f for f in all_files if os.path.getmtime(f) > cutoff_time]
    
    if not recent_files:
        logging.warning("No recent data found! Trying to load ALL data as fallback...")
        recent_files = all_files
        
    if not recent_files:
        raise FileNotFoundError("No CSV files found in data directory.")

    logging.info(f"Found {len(recent_files)} data files to load.")
    
    dfs = []
    for f in recent_files:
        try:

            df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(f, encoding='latin-1', on_bad_lines='skip')
            except:
                logging.warning(f"Skipping unreadable file: {f}")
                continue
        

        age_days = (time.time() - os.path.getmtime(f)) / 86400
        df['__file_age_days'] = age_days
        dfs.append(df)
    
    if not dfs:
        raise ValueError("Could not load any dataframes.")
        
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df: pd.DataFrame, feature_order: dict) -> tuple:
    """
    核心预处理函数：清洗数据、特征工程、去偏见、标准化
    """
    logging.info("Starting data preprocessing...")


    TARGET_MAIN = 'maxdownloadrate_kb'
    

    if TARGET_MAIN in df.columns:
        df[TARGET_MAIN] = pd.to_numeric(df[TARGET_MAIN], errors='coerce').fillna(0)
    else:

        logging.warning(f"Column {TARGET_MAIN} not found, trying fallback...")
        TARGET_MAIN = 'download_mbps' if 'download_mbps' in df.columns else None

    if not TARGET_MAIN:
        raise ValueError("Critical: No valid target column (maxdownloadrate_kb) found!")


    original_rows = len(df)
    

    high_quality_df = df[df[TARGET_MAIN] > 0.1].copy()
    
    use_weight_as_fallback = False
    

    if len(high_quality_df) > 100:
        df = high_quality_df
        logging.info(f"📊 Data Cleaning: {original_rows} -> {len(df)} (Kept rows with real traffic data)")
        y = df[TARGET_MAIN]
    else:

        logging.warning(f"⚠️ Warning: Very few valid speed records ({len(high_quality_df)}).")
        logging.warning("🔄 Fallback Mode Activated: Using 'weight' column as target to prevent crash.")
        
        if 'weight' in df.columns:
            y = df['weight']
            use_weight_as_fallback = True
        else:
            y = df[TARGET_MAIN]


    logging.info("Performing feature engineering...")
    if 'latency' in df.columns:
        df['latency_stability'] = df['latency'] / (df['latency'] + 1e-6)
    

    mask_features = BIASED_FEATURES + COMPLEX_FEATURES
    logging.info(f"Masking biased/complex features: {len(mask_features)} items")
    for col in mask_features:
        if col in df.columns:
            df[col] = 0.0


    ordered_features = [feature_order[i] for i in sorted(feature_order.keys())]
    valid_cols = [col for col in ordered_features if col in df.columns]
    X = df[valid_cols]
    

    X = X.select_dtypes(include=np.number)


    scalers = {}
    

    if CONTINUOUS_FEATURES:
        continuous_present = [c for c in CONTINUOUS_FEATURES if c in X.columns]
        if continuous_present:
            scaler_std = StandardScaler()
            X[continuous_present] = scaler_std.fit_transform(X[continuous_present])
            scalers['standard'] = scaler_std
            scalers['std_features'] = continuous_present
            logging.info(f"Applied StandardScaler to {len(continuous_present)} features")


    if COUNT_FEATURES:
        count_present = [c for c in COUNT_FEATURES if c in X.columns]
        if count_present:
            scaler_robust = RobustScaler()
            X[count_present] = scaler_robust.fit_transform(X[count_present])
            scalers['robust'] = scaler_robust
            scalers['rob_features'] = count_present
            logging.info(f"Applied RobustScaler to {len(count_present)} features")


    logging.info("Calculating sample weights...")
    if '__file_age_days' in df.columns:
        base_weight = 1.0 / (1.0 + 0.1 * df['__file_age_days'])
        
        if not use_weight_as_fallback:
            speed_bonus = np.log1p(df[TARGET_MAIN]) * 0.05
            df['sample_weight'] = base_weight + speed_bonus
        else:
            df['sample_weight'] = base_weight
    else:
        df['sample_weight'] = 1.0
        
    sample_weights = df['sample_weight']

    logging.info(f"Preprocessing complete. Matrix shape: {X.shape}")
    return X, y, sample_weights, scalers

def save_model_and_params(model, scalers, feature_order, output_path: Path):
    """
    保存模型并将归一化参数以 INI 格式追加到文件末尾。
    """
    logging.info(f"Saving model to '{output_path}'...")
    joblib.dump(model, output_path)
    
    feature_name_to_idx = {v: k for k, v in feature_order.items()}

    ini_string = "\n\n[transforms]\n"
    definitions_string = "\n[definitions]\n"
    order_string = "\n[order]\n"
    

    scaler_std = scalers.get('standard')
    std_feature_names = scalers.get('std_features', [])

    if scaler_std and std_feature_names:
        feature_indices = []
        valid_indices = []
        for i, name in enumerate(std_feature_names):
            if name in feature_name_to_idx:
                feature_indices.append(str(feature_name_to_idx[name]))
                valid_indices.append(i)
        
        if feature_indices:
            means = scaler_std.mean_[valid_indices]
            scales = scaler_std.scale_[valid_indices]
            
            definitions_string += "std_type=StandardScaler\n"
            definitions_string += "std_features=" + ",".join(feature_indices) + "\n"
            definitions_string += "std_mean=" + ",".join(f"{x:.6f}" for x in means) + "\n"
            definitions_string += "std_scale=" + ",".join(f"{x:.6f}" for x in scales) + "\n"


    scaler_rob = scalers.get('robust')
    rob_feature_names = scalers.get('rob_features', []) 

    if scaler_rob and rob_feature_names:
        feature_indices = []
        valid_indices = []
        for i, name in enumerate(rob_feature_names):
            if name in feature_name_to_idx:
                feature_indices.append(str(feature_name_to_idx[name]))
                valid_indices.append(i)
        
        if feature_indices:
            centers = scaler_rob.center_[valid_indices]
            scales = scaler_rob.scale_[valid_indices]
            
            definitions_string += "\nrobust_type=RobustScaler\n"
            definitions_string += "robust_features=" + ",".join(feature_indices) + "\n"
            definitions_string += "robust_center=" + ",".join(f"{x:.6f}" for x in centers) + "\n"
            definitions_string += "robust_scale=" + ",".join(f"{x:.6f}" for x in scales) + "\n"


    for i in sorted(feature_order.keys()):
        order_string += f"{i} = {feature_order[i]}\n"

    final_config = ini_string + order_string + definitions_string + "transform=true\n[/transforms]"

    with open(output_path, "ab") as f:
        f.write(final_config.encode('utf-8'))
    
    logging.info("✅ Successfully appended scaling parameters to model file.")

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model for Mihomo.")
    

    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing CSV data files.")
    parser.add_argument("--output", "--output-file", dest="output", type=Path, default=DEFAULT_MODEL_PATH, help="Path to save the trained model.")
    


    
    args = parser.parse_args()


    try:
        go_content = fetch_go_source()
        feature_order = parse_feature_order(go_content)
    except Exception as e:
        logging.error(f"Failed to sync Go source: {e}")
        sys.exit(1)


    try:
        df = load_data(args.data_dir, days=15)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)


    try:
        X, y, weights, scalers = preprocess_data(df, feature_order)
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        sys.exit(1)


    logging.info(f"Training on {len(X)} samples...")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    logging.info("Starting LightGBM training...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=20)
        ]
    )


    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    logging.info(f"🏆 Model Training Complete. Validation MAE: {mae:.4f}")


    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model_and_params(model, scalers, feature_order, args.output)

if __name__ == "__main__":
    main()