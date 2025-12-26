import argparse
import os
import re
import sys
import glob
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Model.bin"
CACHE_DIR = SCRIPT_DIR / "cache"
GO_SOURCE_CACHE_PATH = CACHE_DIR / "transform.go.cache"

GO_SOURCE_URL = "https://raw.githubusercontent.com/vernesong/mihomo/Alpha/component/smart/lightgbm/transform.go"

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

def print_separator(title=None):
    if title:
        print("=" * 60)
        print(f"{title}")
        print("=" * 60)
    else:
        print("=" * 60)

def fetch_go_source():
    print("\n[步骤1] Go 源码解析")
    
    content = ""
    if GO_SOURCE_CACHE_PATH.exists():
        if (time.time() - GO_SOURCE_CACHE_PATH.stat().st_mtime) < 86400:
            print(f"成功加载本地缓存: {GO_SOURCE_CACHE_PATH}")
            return GO_SOURCE_CACHE_PATH.read_text(encoding='utf-8')

    print(f"正在下载 Go 源文件: {GO_SOURCE_URL}")
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        response = requests.get(GO_SOURCE_URL, timeout=10)
        response.raise_for_status()
        content = response.text
        GO_SOURCE_CACHE_PATH.write_text(content, encoding='utf-8')
        print("下载并缓存成功")
        return content
    except Exception as e:
        if GO_SOURCE_CACHE_PATH.exists():
            print(f"下载失败 ({e})，使用旧缓存")
            return GO_SOURCE_CACHE_PATH.read_text(encoding='utf-8')
        raise RuntimeError(f"无法获取 Go 源码: {e}")

def parse_feature_order(go_content):
    print("开始解析 getDefaultFeatureOrder 函数...")
    func_match = re.search(r'func getDefaultFeatureOrder\(\) map\[int\]string \{(.*?)\}', go_content, re.DOTALL)
    if not func_match:
        print("警告: 正则匹配失败，使用后备特征列表")
        return get_fallback_features()

    feature_map = {}
    pairs = re.findall(r'(\d+):\s*"([^"]+)"', func_match.group(1))
    for idx, name in pairs:
        feature_map[int(idx)] = name
    
    if not feature_map:
        return get_fallback_features()
    
    print(f"成功解析 {len(feature_map)} 个特征的顺序定义")
    print(f"特征顺序解析完成，共 {len(feature_map)} 个特征")
    return feature_map

def get_fallback_features():
    features = [
        'success', 'failure', 'connect_time', 'latency', 'upload_mb', 'download_mb', 
        'duration_minutes', 'last_used_seconds', 'is_udp', 'is_tcp', 'asn_feature', 
        'country_feature', 'address_feature', 'port_feature', 'traffic_ratio', 
        'traffic_density', 'connection_type_feature', 'asn_hash', 'host_hash', 
        'ip_hash', 'geoip_hash'
    ]
    return {i: f for i, f in enumerate(features)}

def load_data(data_dir, days=15):
    print("\n[步骤2] 数据加载与清洗")
    print(f"开始从数据目录加载所有 CSV 文件: {data_dir}")
    
    if not data_dir.exists():
        print(f"错误: 目录不存在 {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = glob.glob(str(data_dir / "*.csv"))
    
    cutoff_time = time.time() - (days * 86400)
    recent_files = [f for f in all_files if os.path.getmtime(f) > cutoff_time]
    
    if not recent_files:
        print("警告: 未发现近期数据，尝试加载所有文件...")
        recent_files = all_files
        
    if not recent_files:
        raise FileNotFoundError("没有找到 CSV 文件")

    print(f"--- 找到 {len(recent_files)} 个数据文件 ---")
    
    dfs = []
    for f in recent_files:
        fname = os.path.basename(f)
        print(f"尝试加载文件: {fname}...")
        try:
            df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
            print(f"文件处理完成: {len(df)} 条记录")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(f, encoding='latin-1', on_bad_lines='skip')
                print(f"警告: 文件 '{fname}' 使用 latin-1 编码成功加载: {len(df)} 条记录")
            except:
                print(f"跳过无法读取的文件: {fname}")
                continue
        
        age_days = (time.time() - os.path.getmtime(f)) / 86400
        df['__file_age_days'] = age_days
        dfs.append(df)
    
    if not dfs:
        raise ValueError("无法加载任何数据")
    
    print("\n合并所有数据文件...")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"数据合并完成，总记录数: {len(merged_df)}")
    return merged_df

def preprocess_data(df, feature_order):
    print("\n[步骤3] 特征提取")
    print("开始构建特征矩阵和目标变量...")

    TARGET_MAIN = 'maxdownloadrate_kb'
    if TARGET_MAIN in df.columns:
        df[TARGET_MAIN] = pd.to_numeric(df[TARGET_MAIN], errors='coerce').fillna(0)
    else:
        TARGET_MAIN = 'download_mbps' if 'download_mbps' in df.columns else None

    if not TARGET_MAIN:
        raise ValueError("严重错误: 未找到目标列 (maxdownloadrate_kb)")

    original_rows = len(df)
    high_quality_df = df[df[TARGET_MAIN] > 0.1].copy()
    
    use_weight_as_fallback = False
    
    if len(high_quality_df) > 100:
        df = high_quality_df
        y = df[TARGET_MAIN]
        print(f"数据清洗: {original_rows} -> {len(df)} 条记录 (保留真实测速数据)")
    else:
        print(f"警告: 有效测速数据极少 ({len(high_quality_df)})，启用兜底模式")
        if 'weight' in df.columns:
            y = df['weight']
            use_weight_as_fallback = True
        else:
            y = df[TARGET_MAIN]

    if 'latency' in df.columns:
        df['latency_stability'] = df['latency'] / (df['latency'] + 1e-6)
    
    mask_features = BIASED_FEATURES + COMPLEX_FEATURES
    for col in mask_features:
        if col in df.columns:
            df[col] = 0.0

    ordered_features = [feature_order[i] for i in sorted(feature_order.keys())]
    valid_cols = [col for col in ordered_features if col in df.columns]
    X = df[valid_cols]
    X = X.select_dtypes(include=np.number)

    print(f"特征提取完成 - 特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")

    print("\n[步骤4] 特征标准化")
    print("开始特征标准化处理...")
    
    scalers = {}
    if CONTINUOUS_FEATURES:
        continuous_present = [c for c in CONTINUOUS_FEATURES if c in X.columns]
        if continuous_present:
            scaler_std = StandardScaler()
            X[continuous_present] = scaler_std.fit_transform(X[continuous_present])
            scalers['standard'] = scaler_std
            scalers['std_features'] = continuous_present
            print(f"StandardScaler 处理完成，影响特征数: {len(continuous_present)}")

    if COUNT_FEATURES:
        count_present = [c for c in COUNT_FEATURES if c in X.columns]
        if count_present:
            scaler_robust = RobustScaler()
            X[count_present] = scaler_robust.fit_transform(X[count_present])
            scalers['robust'] = scaler_robust
            scalers['rob_features'] = count_present
            print(f"RobustScaler 处理完成，影响特征数: {len(count_present)}")

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

    return X, y, sample_weights, scalers

def save_model_and_params(model, scalers, feature_order, output_path):
    print("\n[步骤7] 模型保存")
    print(f"开始保存模型至: {output_path}")
    
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
    
    print("模型保存成功，可以直接部署")

def training_logger_cn(period=100):
    def _callback(env):
        if period > 0 and (env.iteration + 1) % period == 0:
            msg_parts = []
            for data_name, eval_name, result, *rest in env.evaluation_result_list:
                if data_name == 'valid_0': d_name = '验证集'
                elif data_name == 'train': d_name = '训练集'
                else: d_name = data_name
                
                if eval_name == 'l1': e_name = 'MAE误差'
                elif eval_name == 'l2': e_name = 'MSE误差'
                else: e_name = eval_name
                
                msg_parts.append(f"{d_name} {e_name}: {result:.6f}")
            print(f"[迭代 {env.iteration + 1:4d}] " + "  ".join(msg_parts))
    _callback.order = 10
    return _callback

def main():
    print_separator("Mihomo 智能权重模型训练")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", "--output-file", dest="output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    try:
        go_content = fetch_go_source()
        feature_order = parse_feature_order(go_content)
    except Exception as e:
        print(f"错误: 同步 Go 源码失败: {e}")
        sys.exit(1)

    try:
        df = load_data(args.data_dir, days=15)
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        sys.exit(1)

    try:
        X, y, weights, scalers = preprocess_data(df, feature_order)
    except Exception as e:
        print(f"错误: 预处理失败: {e}")
        sys.exit(1)

    print("\n[步骤5] 训练测试集划分")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    print(f"数据划分完成 - 训练集: {X_train.shape}, 测试集: {X_val.shape}")

    print("\n[步骤6] 模型训练")
    print("开始 LightGBM 模型训练...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        training_logger_cn(period=100)
    ]

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=callbacks
    )
    print("模型训练完成")

    if model.best_iteration_ == LGBM_PARAMS['n_estimators']:
         print(f"训练状态: 未触发早停。最佳迭代轮数: [{model.best_iteration_}]")
    else:
         print(f"训练状态: 触发早停。最佳迭代轮数: [{model.best_iteration_}]")

    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    
    final_score = max(0, r2 * 10)

    print(f"测试集 MAE误差: {mae:.4f}")
    print(f"模型最终评分: {final_score:.3f} / 10.0")
    
    if final_score > 9.5:
        print("✨ 评级: S级 - 极佳 (规律极强，数据质量完美)")
    elif final_score > 8.0:
        print("🟢 评级: A级 - 良好 (模型可用性高)")
    elif final_score > 6.0:
        print("🟡 评级: B级 - 及格 (部分数据可能存在干扰)")
    elif final_score > 4.0:
        print("🟠 评级: C级 - 一般 (特征关联度弱)")
    else:
        print("🔴 评级: D级 - 不合格 (数据严重不足或噪声过大)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model_and_params(model, scalers, feature_order, args.output)

    print_separator()
    print("模型训练流程完成")
    print(f"输出文件: {args.output}")
    print("模型可进行生产环境部署")
    print_separator()

if __name__ == "__main__":
    main()