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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# --- Constants and Configuration ---

# URL for the Go source file containing feature definitions
GO_SOURCE_URL = "https://raw.githubusercontent.com/vernesong/mihomo/Alpha/component/smart/lightgbm/transform.go"
CACHE_DIR = Path("./cache")
GO_SOURCE_CACHE_PATH = CACHE_DIR / "transform.go.cache"

# Features that create bias ("echo chamber") and should be masked
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

# Features that require log1p transformation (matching Go's prepareFeatures)
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

# Feature types for scaling
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
COUNT_FEATURES = [
    # No count features by default, but can be added here
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


def fetch_go_source() -> str:
    """
    Fetches the Go source code from GitHub or reads from a local cache.

    Returns:
        The content of the Go source file as a string.

    Raises:
        SystemExit: If the download fails and no cache is available.
    """
    logging.info(f"Fetching Go source from {GO_SOURCE_URL}")
    try:
        response = requests.get(GO_SOURCE_URL, timeout=10)
        response.raise_for_status()
        source_code = response.text
        CACHE_DIR.mkdir(exist_ok=True)
        with open(GO_SOURCE_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(source_code)
        logging.info("Successfully fetched and cached Go source.")
        return source_code
    except requests.RequestException as e:
        logging.warning(f"Failed to download Go source: {e}")
        if GO_SOURCE_CACHE_PATH.exists():
            logging.info("Reading from local cache.")
            return GO_SOURCE_CACHE_PATH.read_text(encoding="utf-8")
        else:
            logging.error("No local cache available. Cannot proceed.")
            sys.exit(1)


def parse_feature_order(go_source: str) -> dict[int, str]:
    """
    Parses the getDefaultFeatureOrder function in the Go source to extract
    the feature ID to name mapping.

    Args:
        go_source: The Go source code as a string.

    Returns:
        A dictionary mapping feature integer ID to feature name.

    Raises:
        ValueError: If the feature order function or pattern is not found.
    """
    logging.info("Parsing feature order from Go source...")
    # Regex to find the feature mapping within the function
    pattern = re.compile(
        r'func\s+getDefaultFeatureOrder\(\)\s+map\[int\]string\s+\{.*?return\s+map\[int\]string\s*\{(.*?)\}',
        re.DOTALL
    )
    match = pattern.search(go_source)
    if not match:
        raise ValueError("Could not find 'getDefaultFeatureOrder' function in the Go source.")

    content = match.group(1)
    feature_map = {}
    # Regex for individual map entries, e.g., "0: \"id\","
    entry_pattern = re.compile(r'(\d+)\s*:\s*"([^"]+)"')
    for line in content.split(','):
        if not line.strip():
            continue
        entry_match = entry_pattern.search(line)
        if entry_match:
            feature_id, feature_name = entry_match.groups()
            feature_map[int(feature_id)] = feature_name
    
    if not feature_map:
        raise ValueError("Failed to parse any features from the Go source.")

    logging.info(f"Successfully parsed {len(feature_map)} features.")
    return feature_map


def fetch_training_data(data_dir: Path) -> Path:
    """
    Placeholder function to fetch training data from a remote source.

    In a real-world scenario, this function would download data from a
    cloud storage (like S3), a database, or a dedicated API.

    For this example, it simulates the process by creating a dummy CSV file.

    Args:
        data_dir: The local directory to store the fetched data.

    Returns:
        The path to the directory containing the data.
    """
    logging.info(f"Simulating data fetch to '{data_dir}'...")
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
        
    logging.info("Dummy data has been created.")
    return data_dir


def load_data(data_dir: Path, time_window_days: int = 15) -> pd.DataFrame:
    """
    Loads CSV files from the specified directory within a time window.

    Args:
        data_dir: The directory containing CSV data files.
        time_window_days: The number of recent days to load data from.

    Returns:
        A pandas DataFrame containing the combined data.
    """
    logging.info(f"Loading data from the last {time_window_days} days from '{data_dir}'...")
    all_files = list(data_dir.glob("*.csv"))
    if not all_files:
        logging.warning(f"No CSV files found in directory: {data_dir}")
        return pd.DataFrame()

    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=time_window_days)
    recent_files = []
    for f in all_files:
        mod_time = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        if mod_time >= cutoff_date:
            recent_files.append(f)

    if not recent_files:
        logging.warning("No files found within the time window.")
        return pd.DataFrame()

    logging.info(f"Found {len(recent_files)} recent data files to load.")

    df_list = []
    for f in recent_files:
        # Try different encodings
        for encoding in ['utf-8', 'gbk', 'latin-1']:
            try:
                df = pd.read_csv(f, encoding=encoding)
                df['file_age_days'] = (datetime.datetime.now() - datetime.datetime.fromtimestamp(f.stat().st_mtime)).days
                df_list.append(df)
                logging.info(f"Loaded '{f.name}' with {encoding} encoding.")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            logging.warning(f"Could not decode or parse file: {f.name}")

    if not df_list:
        logging.error("Failed to load any data.")
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def preprocess_data(df: pd.DataFrame, feature_order: dict) -> tuple:
    """
    Preprocesses the raw data for model training.

    Args:
        df: The input DataFrame.
        feature_order: The feature order map from Go source.

    Returns:
        A tuple containing:
        - X (features)
        - y_combined (a DataFrame with main and auxiliary targets)
        - sample_weights
        - scalers (dictionary of fitted scalers)
    """
    logging.info("Starting data preprocessing...")

    # --- Feature Engineering ---
    logging.info("Performing feature engineering...")
    # Handle potential division by zero
    df['latency_stability'] = df['latency_avg'] / (df['latency_min'] + 1e-6)
    df['connection_efficiency'] = df['success_rate'] / (df['connect_time'] + 1e-6)
    logging.info("Engineered features 'latency_stability' and 'connection_efficiency'.")


    # Set target variables
    TARGET_MAIN = 'download_mbps'

    if TARGET_MAIN not in df.columns:
        # Fallback for older data formats or dummy data
        TARGET_MAIN = 'maxdownloadrate_kb' if 'maxdownloadrate_kb' in df.columns else None
        if not TARGET_MAIN:
            raise ValueError("Main target variable ('download_mbps' or 'maxdownloadrate_kb') not found in data.")
        logging.warning(f"Main target 'download_mbps' not found, falling back to '{TARGET_MAIN}'.")
    
    y = df[TARGET_MAIN]

    # Mask biased features
    logging.info(f"Masking biased features: {BIASED_FEATURES}")
    for col in BIASED_FEATURES:
        if col in df.columns:
            df[col] = 0.0

    # Align feature columns with the order from Go source
    ordered_features = [feature_order[i] for i in sorted(feature_order.keys())]
    # Keep only features present in the dataframe
    X = df[[col for col in ordered_features if col in df.columns]]

    # Drop non-numeric and identifier columns that are not features
    X = X.select_dtypes(include=np.number)

    # Apply Log1p transformation to match Go's prepareFeatures
    logging.info(f"Applying Log1p transformation to: {LOG1P_FEATURES}")
    for col in LOG1P_FEATURES:
        if col in X.columns:
            # Clip negative values to 0 before log1p to avoid errors
            X[col] = np.log1p(X[col].clip(lower=0))

    # Apply scaling
    scalers = {}
    if CONTINUOUS_FEATURES:
        continuous_present = [c for c in CONTINUOUS_FEATURES if c in X.columns]
        # Use Standard Scaler for all continuous features to match Go's expectation
        # Go's transform.go seems to default to StandardScaler or RobustScaler
        # We will use StandardScaler as the primary scaler for now.
        scaler_std = StandardScaler()
        X[continuous_present] = scaler_std.fit_transform(X[continuous_present])
        scalers['standard'] = scaler_std
        scalers['std_features'] = continuous_present # Save feature names for ini generation
        logging.info(f"Applied StandardScaler to: {continuous_present}")


    if COUNT_FEATURES:
        count_present = [c for c in COUNT_FEATURES if c in X.columns]
        scaler_robust = RobustScaler()
        X[count_present] = scaler_robust.fit_transform(X[count_present])
        scalers['robust'] = scaler_robust
        scalers['robust_features'] = count_present # Save feature names for ini generation
        logging.info(f"Applied RobustScaler to: {count_present}")


    # Calculate sample weights with time decay
    logging.info("Calculating sample weights with time decay...")
    df['sample_weight'] = 1 / (1 + 0.1 * df['file_age_days'])
    sample_weights = df['sample_weight']

    logging.info("Preprocessing complete.")
    return X, y, sample_weights, scalers


def save_model_and_params(model, scalers, feature_order, output_path: Path):
    """
    Saves the LightGBM model and appends the scaling parameters in INI format.
    Matching the format expected by mihomo-Alpha/component/smart/lightgbm/transform.go

    Args:
        model: The trained LightGBM model.
        scalers: Dictionary of fitted scalers.
        feature_order: The feature order map.
        output_path: The path to save the final model file.
    """
    logging.info(f"Saving model to '{output_path}'...")
    joblib.dump(model, output_path)
    logging.info("Model saved successfully.")

    # Invert feature order to map Name -> Index
    feature_name_to_idx = {v: k for k, v in feature_order.items()}

    # --- Construct INI configuration string ---
    ini_string = "\n\n[transforms]\n"
    definitions_string = "\n[definitions]\n"
    order_string = "\n[order]\n"
    
    scaler_std = scalers.get('standard')
    std_feature_names = scalers.get('std_features', [])

    if scaler_std and std_feature_names:
        # Get indices for the features
        feature_indices = []
        valid_indices = []
        
        for i, name in enumerate(std_feature_names):
            if name in feature_name_to_idx:
                feature_indices.append(str(feature_name_to_idx[name]))
                valid_indices.append(i)
            else:
                logging.warning(f"Feature {name} not found in feature order map, skipping in transform config.")

        if feature_indices:
            # Filter mean and scale to only include valid features
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
            else:
                logging.warning(f"Feature {name} not found in feature order map, skipping in transform config.")

        if feature_indices:
            centers = scaler_robust.center_[valid_indices]
            scales = scaler_robust.scale_[valid_indices]

            definitions_string += "\nrobust_type=RobustScaler\n"
            definitions_string += "robust_features=" + ",".join(feature_indices) + "\n"
            definitions_string += "robust_center=" + ",".join(f"{x:.6f}" for x in centers) + "\n"
            definitions_string += "robust_scale=" + ",".join(f"{x:.6f}" for x in scales) + "\n"

    # Build [order] section
    for i in sorted(feature_order.keys()):
        order_string += f"{i} = {feature_order[i]}\n"


    final_config = ini_string + order_string + definitions_string + "transform=true\n[/transforms]"

    # Append to the model file
    with open(output_path, "ab") as f:
        f.write(final_config.encode('utf-8'))
    
    logging.info("Successfully appended scaling parameters to model file.")


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Evaluates the model on the main target and returns its MAE."""
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    return mae

def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="Mihomo Smart Node Trainer")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Directory for training data.")
    parser.add_argument("--output-file", type=Path, default=Path("./Model.bin"), help="Path to save the model.")
    parser.add_argument("--days", type=int, default=15, help="Number of recent days of data to use.")
    parser.add_argument("--champion-model-path", type=Path, default=None, help="Path to the champion model for comparison.")
    args = parser.parse_args()

    # --- Pipeline ---
    go_source = fetch_go_source()
    feature_order = parse_feature_order(go_source)
    fetched_data_dir = fetch_training_data(args.data_dir)
    df = load_data(fetched_data_dir, args.days)
    if df.empty:
        logging.error("No data loaded. Exiting.")
        sys.exit(1)

    X, y, sample_weights, scalers = preprocess_data(df, feature_order)

    X_train, X_val, y_train, y_val, weights_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test, weights_train, _ = train_test_split(
        X_train, y_train, weights_train, test_size=0.2, random_state=42
    )

    logging.info("Training Challenger LightGBM model (Regression L1)...")
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
    logging.info("Challenger model training complete.")

    challenger_mae = evaluate_model(challenger_model, X_val, y_val)
    logging.info(f"Challenger model validation MAE on main target: {challenger_mae:.4f}")

    new_model_is_champion = True
    if args.champion_model_path and args.champion_model_path.exists():
        logging.info(f"Loading champion model from {args.champion_model_path} for comparison.")
        try:
            champion_model = joblib.load(args.champion_model_path)
            champion_mae = evaluate_model(champion_model, X_val, y_val)
            logging.info(f"Champion model validation MAE on main target: {champion_mae:.4f}")

            if challenger_mae >= champion_mae:
                new_model_is_champion = False
                logging.info(f"Challenger is not better than champion. Keeping the old model.")
            else:
                logging.info("Challenger is better than champion. Proceeding to save new model.")
        except Exception as e:
            logging.warning(f"Could not load or evaluate champion model: {e}. Proceeding as if no champion exists.")
    else:
        logging.info("No champion model found. The new model will become the champion.")

    print(f"::set-output name=new_model_is_champion::{str(new_model_is_champion).lower()}")

    if new_model_is_champion:
        save_model_and_params(challenger_model, scalers, feature_order, args.output_file)

    logging.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
