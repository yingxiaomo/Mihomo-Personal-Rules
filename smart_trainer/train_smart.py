import argparse
import datetime
import logging
import os
import re
import sys
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

GO_SOURCE_URL = "https://raw.githubusercontent.com/vernesong/mihomo/Alpha/component/smart/lightgbm/transform.go"
CACHE_DIR = Path("./cache")
GO_SOURCE_CACHE_PATH = CACHE_DIR / "transform.go.cache"

BIASED_FEATURES = [
    'download_mb',
    'upload_mb',
    'traffic_density',
    'traffic_ratio',
    'last_used_seconds',
    'history_download_mb',
    'history_upload_mb',
    'duration_minutes',
    'success',
    'failure',
]

LOG1P_FEATURES = [
    'connect_time',
    'latency',
    'upload_mb',
    'history_upload_mb',
    'maxuploadrate_kb',
    'history_maxuploadrate_kb',
    'download_mb',
    'history_download_mb',
    'maxdownloadrate_kb',
    'history_maxdownloadrate_kb',
    'duration_minutes',
    'last_used_seconds',
]

CONTINUOUS_FEATURES = [
    'success',
    'failure',
    'connect_time',
    'latency',
    'upload_mb',
    'history_upload_mb',
    'maxuploadrate_kb',
    'history_maxuploadrate_kb',
    'download_mb',
    'history_download_mb',
    'maxdownloadrate_kb',
    'history_maxdownloadrate_kb',
    'duration_minutes',
    'last_used_seconds',
    'traffic_density',
    'traffic_ratio',
    'asn_hash',
    'host_hash',
    'ip_hash',
    'geoip_hash',
]
COUNT_FEATURES = []

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)

def fetch_go_source() -> str:
    logging.info("[步骤1] Go 源码解析")
    logging.info(f"正在从 {GO_SOURCE_URL} 获取 Go 源码...")
    try:
        response = requests.get(GO_SOURCE_URL, timeout=10)
        response.raise_for_status()
        source_code = response.text
        CACHE_DIR.mkdir(exist_ok=True)
        with open(GO_SOURCE_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(source_code)
        logging.info("成功获取并缓存 Go 源码")
        return source_code
    except requests.RequestException as e:
        logging.warning(f"下载 Go 源码失败: {e}")
        if GO_SOURCE_CACHE_PATH.exists():
            logging.info("正在从本地缓存读取")
            return GO_SOURCE_CACHE_PATH.read_text(encoding="utf-8")
        else:
            logging.error("无本地缓存可用，程序退出")
            sys.exit(1)

def parse_feature_order(go_source: str) -> dict[int, str]:
    logging.info("开始解析 getDefaultFeatureOrder 函数...")
    pattern = re.compile(
        r'func\s+getDefaultFeatureOrder\(\)\s+map\[int\]string\s+\{.*?return\s+map\[int\]string\s*\{(.*?)\}',
        re.DOTALL
    )
    match = pattern.search(go_source)
    if not match:
        raise ValueError("在 Go 源码中未找到 'getDefaultFeatureOrder' 函数")
    content = match.group(1)
    feature_map = {}
    entry_pattern = re.compile(r'(\d+)\s*:\s*"([^"]+)"')
    for line in content.split(','):
        if not line.strip():
            continue
        entry_match = entry_pattern.search(line)
        if entry_match:
            feature_id, feature_name = entry_match.groups()
            feature_map[int(feature_id)] = feature_name
    if not feature_map:
        raise ValueError("解析特征失败")
    logging.info(f"成功解析 {len(feature_map)} 个特征的顺序定义")
    return feature_map

def fetch_training_data(data_dir: Path) -> Path:
    logging.info(f"模拟数据获取至 '{data_dir}'...")
    data_dir.mkdir(exist_ok=True)
    dummy_csv_path = data_dir / "dummy_data.csv"
    dummy_data = (
        '"id","node_id","node_group","node_type","success_rate","upload_mb","download_mbps","download_mb","latency_avg","latency_min","failure_rate","traffic_density","traffic_ratio","last_used_seconds"\n'
        '1,"node_a","HK","Vmess",1.0,10.5,150.0,2048.0,80.0,75.0,0.0,0.8,0.5,1800\n'
        '2,"node_b","SG","Trojan",0.98,5.2,80.0,1024.0,120.0,110.0,0.02,0.6,0.3,3600\n'
        '3,"node_c","JP","Vmess",0.99,20.0,200.0,4096.0,60.0,55.0,0.01,0.9,0.7,900\n'
    )
    with open(dummy_csv_path, "w", encoding="utf-8") as f:
        f.write(dummy_data)
    logging.info("虚拟数据已创建")
    return data_dir

def load_data(data_dir: Path, time_window_days: int = 15) -> pd.DataFrame:
    logging.info("\n[步骤2] 数据加载与清洗")
    logging.info(f"开始从数据目录加载最近 {time_window_days} 天的 CSV 文件: {data_dir}")
    all_files = list(data_dir.glob("*.csv"))
    if not all_files:
        logging.warning(f"目录中未找到 CSV 文件: {data_dir}")
        return pd.DataFrame()
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=time_window_days)
    recent_files = []
    for f in all_files:
        mod_time = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        if mod_time >= cutoff_date:
            recent_files.append(f)
    if not recent_files:
        logging.warning("时间窗口内未找到文件")
        return pd.DataFrame()
    logging.info(f"--- 找到 {len(recent_files)} 个数据文件 ---")
    df_list = []
    for f in recent_files:
        logging.info(f"尝试加载文件: {f.name}...")
        for encoding in ['utf-8', 'gbk', 'latin-1']:
            try:
                df = pd.read_csv(f, encoding=encoding)
                df['file_age_days'] = (datetime.datetime.now() - datetime.datetime.fromtimestamp(f.stat().st_mtime)).days
                if not df.empty:
                    df_list.append(df)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            logging.warning(f"无法解码或解析文件: {f.name}")
    if not df_list:
        logging.error("未能加载任何有效数据")
        return pd.DataFrame()
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def preprocess_data(df: pd.DataFrame, feature_order: dict) -> tuple:
    logging.info("\n[步骤3] 特征提取与废数据过滤")
    logging.info("开始构建特征矩阵和目标变量...")
    df['latency_stability'] = df['latency_avg'] / (df['latency_min'] + 1e-6)
    df['connection_efficiency'] = df['success_rate'] / (df['connect_time'] + 1e-6)
    TARGET_MAIN = 'download_mbps'
    if TARGET_MAIN not in df.columns:
        TARGET_MAIN = 'maxdownloadrate_kb'
    initial_rows = len(df)
    df[TARGET_MAIN] = pd.to_numeric(df[TARGET_MAIN], errors='coerce')
    df = df[df[TARGET_MAIN] > 0]
    df = df[np.isfinite(df[TARGET_MAIN])]
    final_rows = len(df)
    logging.info(f"已过滤 {initial_rows - final_rows} 条废数据 (NaN或0速度). 剩余有效记录数: {final_rows}")
    if df.empty:
        raise ValueError("数据清洗后为空，无法继续训练")
    y = df[TARGET_MAIN]
    logging.info(f"屏蔽偏见特征: {BIASED_FEATURES}")
    for col in BIASED_FEATURES:
        if col in df.columns:
            df[col] = 0.0
    ordered_features = [feature_order[i] for i in sorted(feature_order.keys())]
    X = df[[col for col in ordered_features if col in df.columns]]
    X = X.select_dtypes(include=np.number)
    X = X.fillna(0)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    logging.info(f"应用 Log1p 变换: {LOG1P_FEATURES}")
    for col in LOG1P_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))
    logging.info(f"特征提取完成 - 特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")
    logging.info("\n[步骤4] 特征标准化")
    scalers = {}
    if CONTINUOUS_FEATURES:
        continuous_present = [c for c in CONTINUOUS_FEATURES if c in X.columns]
        scaler_std = StandardScaler()
        X[continuous_present] = scaler_std.fit_transform(X[continuous_present])
        scalers['standard'] = scaler_std
        scalers['std_features'] = continuous_present
        logging.info(f"StandardScaler 处理完成，影响特征数: {len(continuous_present)}")
    if COUNT_FEATURES:
        count_present = [c for c in COUNT_FEATURES if c in X.columns]
        scaler_robust = RobustScaler()
        X[count_present] = scaler_robust.fit_transform(X[count_present])
        scalers['robust'] = scaler_robust
        scalers['robust_features'] = count_present
        logging.info(f"RobustScaler 处理完成，影响特征数: {len(count_present)}")
    df['sample_weight'] = 1 / (1 + 0.1 * df['file_age_days'])
    sample_weights = df['sample_weight']
    return X, y, sample_weights, scalers

def save_model_and_params(model, scalers, feature_order, output_path: Path):
    logging.info("\n[步骤7] 模型保存")
    logging.info(f"开始保存模型至: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    scaler_robust = scalers.get('robust')
    robust_feature_names = scalers.get('robust_features', [])
    if scaler_robust and robust_feature_names:
        feature_indices = []
        valid_indices = []
        for i, name in enumerate(robust_feature_names):
            if name in feature_name_to_idx:
                feature_indices.append(str(feature_name_to_idx[name]))
                valid_indices.append(i)
        if feature_indices:
            centers = scaler_robust.center_[valid_indices]
            scales = scaler_robust.scale_[valid_indices]
            definitions_string += "\nrobust_type=RobustScaler\n"
            definitions_string += "robust_features=" + ",".join(feature_indices) + "\n"
            definitions_string += "robust_center=" + ",".join(f"{x:.6f}" for x in centers) + "\n"
            definitions_string += "robust_scale=" + ",".join(f"{x:.6f}" for x in scales) + "\n"
    for i in sorted(feature_order.keys()):
        order_string += f"{i} = {feature_order[i]}\n"
    final_config = ini_string + order_string + definitions_string + "transform=true\n[/transforms]"
    with open(output_path, "ab") as f:
        f.write(final_config.encode('utf-8'))
    logging.info("模型保存成功，可以直接部署")
    logging.info("============================================================")
    logging.info("模型训练流程完成")
    logging.info(f"输出文件: {output_path}")
    logging.info("============================================================")

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> tuple:
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    return mae, r2

def main():
    print("============================================================")
    print("Mihomo 智能权重模型训练")
    print("============================================================")
    print("")
    parser = argparse.ArgumentParser(description="Mihomo Smart Node Trainer")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Directory for training data.")
    parser.add_argument("--output-file", type=Path, default=Path("./Model.bin"), help="Path to save the model.")
    parser.add_argument("--days", type=int, default=15, help="Number of recent days of data to use.")
    parser.add_argument("--champion-model-path", type=Path, default=None, help="Path to the champion model for comparison.")
    args = parser.parse_args()
    go_source = fetch_go_source()
    feature_order = parse_feature_order(go_source)
    fetched_data_dir = fetch_training_data(args.data_dir)
    df = load_data(fetched_data_dir, args.days)
    if df.empty:
        logging.error("没有数据加载，程序退出")
        sys.exit(1)
    X, y, sample_weights, scalers = preprocess_data(df, feature_order)
    logging.info("\n[步骤5] 训练测试集划分")
    X_train, X_val, y_train, y_val, weights_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test, weights_train, _ = train_test_split(
        X_train, y_train, weights_train, test_size=0.2, random_state=42
    )
    logging.info(f"数据划分完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    logging.info("\n[步骤6] 模型训练")
    logging.info("开始 LightGBM 模型训练...")
    challenger_model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    challenger_model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )
    logging.info("模型训练完成")
    challenger_mae, challenger_r2 = evaluate_model(challenger_model, X_val, y_val)
    logging.info(f"训练集R²得分: {challenger_r2:.4f} (MAE: {challenger_mae:.4f})")
    new_model_is_champion = True
    if args.champion_model_path and args.champion_model_path.exists():
        logging.info(f"正在加载冠军模型进行对比: {args.champion_model_path}")
        try:
            champion_model = joblib.load(args.champion_model_path)
            champion_mae, champion_r2 = evaluate_model(champion_model, X_val, y_val)
            logging.info(f"冠军模型 R²: {champion_r2:.4f} (MAE: {champion_mae:.4f})")
            if challenger_mae >= champion_mae:
                new_model_is_champion = False
                logging.info("新模型表现未超过旧模型，保留旧模型")
            else:
                logging.info("新模型表现更优，准备替换")
        except Exception as e:
            logging.warning(f"无法加载或评估冠军模型: {e}，将直接使用新模型")
    else:
        logging.info("未找到旧模型，新模型将直接上位")
    print(f"::set-output name=new_model_is_champion::{str(new_model_is_champion).lower()}")
    if new_model_is_champion:
        save_model_and_params(challenger_model, scalers, feature_order, args.output_file)
if __name__ == "__main__":
    main()