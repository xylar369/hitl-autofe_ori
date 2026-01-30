from utils.data import download_and_save_dataset, train_test_dataset_build, load_dataset_from_csv, minmax_scale_datasets, csv_processing, get_data_domains, mask_feature_name, get_columns_with_inf
from utils.prompt import get_prompt, extract_and_save_code, get_init_prompt, get_prompt_v2, get_prompt_v3, get_prompt_v4, get_prompt_shap
from utils.xgboost_train import train_best_xgboost_model, train_xgboost_model
from utils.run_code import run_feature_engineering, safe_apply_feature_engineering
from utils.config_loader import load_config, get_config_or_default, validate_config, save_config
from utils.importance_compute import process_shap_importance
import shap
import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
from openai import OpenAI
import json
import time
import argparse
import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def quick_eval(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    使用 5-Fold 交叉验证评估特征稳定性。
    返回: score (mean - std), average_test_acc
    """
    try:
        # 1. 合并训练集和验证集，用于交叉验证
        X_combined = pd.concat([x_train, x_val], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
        # 2. 初始化 5 折分层采样
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        val_acc_list = []
        test_acc_list = []
        
        # 3. 开始交叉验证迭代
        for train_idx, val_idx in skf.split(X_combined, y_combined):
            # 划分当前的 Fold
            xtr_fold = X_combined.iloc[train_idx]
            ytr_fold = y_combined.iloc[train_idx]
            xv_fold = X_combined.iloc[val_idx]
            yv_fold = y_combined.iloc[val_idx]
            
            # 缩放数据 (使用你原有的 minmax 逻辑)
            # 注意：每个 Fold 都要重新 Scale 以防止数据泄露
            xtr_s, xv_s, xte_s = minmax_scale_datasets(xtr_fold, xv_fold, x_test)
            
            # 训练模型
            # train_xgboost_model 返回: clf, train_acc, val_acc, test_acc
            _, _, v_acc, te_acc = train_xgboost_model(
                xtr_s, ytr_fold, xv_s, yv_fold, xte_s, y_test, params={}
            )
            
            val_acc_list.append(v_acc)
            test_acc_list.append(te_acc)
            
        # 4. 计算统计量
        val_mean = np.mean(val_acc_list)
        val_std = np.std(val_acc_list)
        avg_test_acc = np.mean(test_acc_list)
        
        # 5. 计算最终得分 (均值减去标准差)
        # 这种得分方式会惩罚那些在不同 Fold 之间波动巨大的特征
        final_score = val_mean - val_std
        
        print(f"CV Results: Mean={val_mean:.4f}, Std={val_std:.4f}, Score={final_score:.4f}")
        
        return final_score, avg_test_acc

    except Exception as e:
        print(f"Eval failed during CV: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0

# --- Helper: Check for Infs ---
# def check_inf_error(df, feature_name):
#     if feature_name and feature_name in df.columns:
#         if np.isinf(df[feature_name]).any():
#             return True
#     return False

def check_inf_error(df, feature_name):
    """
    1. Converts the column to numeric INPLACE.
       - Tries to convert valid numbers (e.g., category "1.5" -> float 1.5).
       - If that fails (e.g., category "High"), it converts to integer codes.
    2. Checks for infinite values.
    """
    if feature_name not in df.columns:
        return False

    col = df[feature_name]

    # Check if the column is not already numeric (e.g., category or object)
    if pd.api.types.is_categorical_dtype(col) or pd.api.types.is_object_dtype(col):
        
        # Attempt 1: Try strict numeric conversion (good for "1", "2.5" stored as text)
        numeric_converted = pd.to_numeric(col, errors='coerce')
        
        # If conversion worked (we didn't turn everything into NaN), use it
        if not numeric_converted.isna().all():
            df[feature_name] = numeric_converted
        
        # Attempt 2: If it was actual text (e.g. "Male", "Female"), encode as Integers
        else:
            # Convert to category type first (if object), then extract codes
            df[feature_name] = col.astype('category').cat.codes
            
            # Note: .cat.codes turns NaNs into -1. If you strictly need NaNs, handle separately.

    # Now the column is definitely numeric in 'df', so we can safely check for inf
    if np.isinf(df[feature_name]).any():
        return True

    return False

def add_noise_column_at_last(train_data, val_data, test_data, noise_level=0.1):
    """
    Adds a noise column to the end of the dataset.

    Parameters:
    train_data (pd.DataFrame): Training dataset.
    val_data (pd.DataFrame): Validation dataset.
    test_data (pd.DataFrame): Test dataset.
    noise_level (float): Standard deviation of the Gaussian noise to be added.

    Returns:
    pd.DataFrame, pd.DataFrame, pd.DataFrame: Datasets with added noise column.
    """
    for data in [train_data, val_data, test_data]:
        noise = np.random.normal(0, noise_level, size=(data.shape[0], 1))
        data['noise'] = noise
    return train_data, val_data, test_data



"""
In this version, I want to add the past best strategies for prompt building.
I want to obtain 3 parts information.
1. past optimal stratgies.
2. improvement.
3. validation scores.
"""
def processing_from_csv(csv_train, csv_val, csv_test, metadata_path, steps=10, masked_flag=False,
                        results_dir="./results/", prompt_dir="./prompt/", code_dir="./code/", llm_model="gpt-5-nano",
                        temperature=0.2, drop_threshold=1.5, port=8000):

    print("Start Processing (Optimization Target: Test Accuracy)")
    
    # init log
    os.makedirs(results_dir, exist_ok=True)
    itr_history_resuts_save_path = os.path.join(results_dir, "val_test.log")
    logging.basicConfig(
        filename = itr_history_resuts_save_path,
        filemode = 'w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO            
    )

    # 1. Load Data
    train_df = pd.read_csv(csv_train)
    val_df = pd.read_csv(csv_val)
    test_df = pd.read_csv(csv_test)

    # 2. Pre-processing (Masking) - BEFORE combining
    if masked_flag:
        train_df = mask_feature_name(train_df)
        val_df = mask_feature_name(val_df)
        test_df = mask_feature_name(test_df)

    # 4. CSV Processing
    x_train_df, y_train = csv_processing(train_df)
    x_val_df, y_val = csv_processing(val_df)
    x_test_df, y_test = csv_processing(test_df)

    # Load Metadata
    dataset_name = "unknown"
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
        dataset_name = meta.get("dataset_name", "unknown")
        description = meta.get("description", "")
    f.close()

    # description_path = metadata_path.replace("metadata.json", f"{dataset_name}_description.txt")
    # description = ""
    # with open(description_path, "r") as f:
    #     description = f.read()
    # f.close()
    
    # Init Tracking
    last_step_report = "No previous strategies. This is the first iteration."
    
    # Initial Baseline
    print("Running Baseline on Combined Train vs Test...")
    x_tr, x_v, x_te = minmax_scale_datasets(x_train_df, x_val_df, x_test_df)
    
    # train_xgboost returns: clf, train_acc, val_acc, test_acc
    # Since x_v == x_te, val_acc will be equal to test_acc
    start_time = time.time()
    _, _, val_acc, test_acc = train_xgboost_model(x_tr, y_train, x_v, y_val, x_te, y_test, params={})
    end_time = time.time()
    print(f"XGBoost Model Training Time: {end_time - start_time}")
    # Global Best Tracking
    # current_best_val is now tracking Test Accuracy because val_df is test_df
    global_best_val = val_acc  # 历史最高的 validation 准确率（永不回退）
    current_best_val = val_acc  # 当前状态的 validation 准确率（可能探索性下降）
    baseline_result = test_acc 
    best_test_result = test_acc  # Track the test_acc corresponding to the highest val_acc
    best_val_acc = val_acc  # Track the highest val_acc
    
    print(f"Baseline Test Accuracy: {current_best_val:.4f}")

    importance_df = pd.DataFrame()
    useful_features = []
    useless_features = []
    noise_threshold = 0
    train_val_domain = {}
    previous_error_generated_faeture = ""
    current_step = 0
    
    # ===== NEW: Track past best strategies =====
    past_strategies_history = []  # List to store all strategies and their results
    baseline_strategy = {
        "step": -1,
        "strategy_name": "Baseline",
        "validation_score": baseline_result,
        "improvement": 0.0,
        "best_score_at_step": baseline_result,
        "description": "Initial baseline model without any feature engineering"
    }
    past_strategies_history.append(baseline_strategy)

    while current_step < steps:
        print(f"\n=== Step {current_step} ===")
        
        # 1. Feature Importance Analysis
        try:
            start_time = time.time()
            train_val_domain = get_data_domains(x_train_df)
            x_tr, x_v, x_te = minmax_scale_datasets(x_train_df, x_val_df, x_test_df)
            x_cor_train, x_cor_val, x_cor_test = add_noise_column_at_last(x_tr, x_v, x_te, noise_level=0.1)
            # Train model with noise on all datasets for consistency
            analysis_clf, _, _, _ = train_xgboost_model(x_cor_train, y_train, x_cor_val, y_val, x_cor_test, y_test, params={})
            # Compute SHAP importance on validation set with noise
            print(f"DEBUG: x_cor_val shape: {x_cor_val.shape}, columns: {x_cor_val.columns.tolist()}")
            importance_df, useful_features, useless_f, _, noise_threshold = process_shap_importance(analysis_clf, x_cor_val)
            print(f"DEBUG: importance_df shape: {importance_df.shape}, columns: {importance_df.columns.tolist()}")
            useless_features = list(set(useless_features + useless_f))
            end_time = time.time()
            print(f"Feature Improtance Analysis Time: {end_time - start_time}")
            # useless_features = useless_f
        except Exception as e:
            print(f"Analysis Warning: {e}")
            import traceback
            traceback.print_exc()
            # Skip this step if feature importance analysis fails
            current_step += 1
            continue

        # 2. Build Prompt
        prompt = ''
        if current_step == 0:
            prompt = get_prompt_shap(
                "classification", dataset_name, description, current_best_val, importance_df, masked_flag,
                useful_features, useless_features, noise_threshold, 
                last_step_report="None (First Step)", domain_dict=train_val_domain,
                past_strategies_history=past_strategies_history
            )
        else:
            prompt = get_prompt_shap(
                "classification", dataset_name, description, current_best_val, importance_df, masked_flag,
                useful_features, useless_features, noise_threshold, 
                last_step_report=last_step_report, domain_dict=train_val_domain,
                past_strategies_history=past_strategies_history
            )

        # 3. Call LLM
        start_time = time.time()
        client = None
        if "gpt" in llm_model:
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a Senior Data Scientist. You shold improve the model from features' SHAP importance."},
                {"role": "user", "content": prompt}
            ],
        )
        else:
            client = OpenAI(
                base_url=f"http://localhost:{str(port)}/v1",
                api_key="Empty",
                timeout=60.0
            )
            response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a Senior Data Scientist."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        
        generated_text = response.choices[0].message.content
        end_time = time.time()
        print(f"Response Generating Time For LLM: {end_time-start_time}")

        # Save Prompt/Response
        os.makedirs(prompt_dir, exist_ok=True)
        with open(f"{prompt_dir}/{dataset_name}_prompt_response_step{current_step}.txt", "w") as f:
            f.write("### PROMPT ###\n")
            f.write(prompt)
            f.write("\n\n### RESPONSE ###\n")
            f.write(generated_text)
        
        # Save Code
        os.makedirs(code_dir, exist_ok=True)
        code_file = f"{code_dir}/{dataset_name}_step{current_step}.py"        
        extract_and_save_code(generated_text, code_file)

        # 4. === CHAMPION SELECTION (Ablation Study) ===
        start_time = time.time()
        # Note: We pass the combined train and the test set (as val) here
        xtrain_mix, xval_mix, xtest_mix, dropped_cols, gen_col, error = safe_apply_feature_engineering(
            code_file, x_train_df, x_val_df, x_test_df
        )

        step_log = []
        winner_name = "Original"
        
        # Default: Keep Original
        next_xtrain, next_xval, next_xtest = x_train_df, x_val_df, x_test_df

        if error:
            decision_text = f"FAILED (Error: {error}). Kept Original."
            last_step_report = f"Last Strategy: {decision_text}"
        else:
            # Check Inf (with the fix to auto-convert categories)
            inf_error = False
            if gen_col and check_inf_error(xtrain_mix, gen_col):
                inf_error = True
                step_log.append(f"Generated feature '{gen_col}' contained Infinity. Discarding generation.")
                useless_features.append(gen_col)

            candidates = []

            # Candidate 1: DROP ONLY
            actual_dropped = [c for c in dropped_cols if c in x_train_df.columns]
            if actual_dropped:
                xt_drop = x_train_df.drop(columns=actual_dropped)
                xv_drop = x_val_df.drop(columns=actual_dropped)
                xte_drop = x_test_df.drop(columns=actual_dropped)
                # quick_eval returns accuracy on 'xv_drop', which is the Test set
                score_drop, test_drop = quick_eval(xt_drop, y_train, xv_drop, y_val, xte_drop, y_test)
                candidates.append(("Drop Only", score_drop, test_drop, xt_drop, xv_drop, xte_drop))
                step_log.append(f"- Drop Strategy ({actual_dropped}): {score_drop:.4f}")

            # Candidate 2: GENERATE ONLY
            if gen_col and not inf_error:
                xt_gen = x_train_df.copy(); xt_gen[gen_col] = xtrain_mix[gen_col]
                xv_gen = x_val_df.copy(); xv_gen[gen_col] = xval_mix[gen_col]
                xte_gen = x_test_df.copy(); xte_gen[gen_col] = xtest_mix[gen_col]
                # quick_eval on Test
                score_gen, test_gen = quick_eval(xt_gen, y_train, xv_gen, y_val, xte_gen, y_test)
                candidates.append(("Generate Only", score_gen, test_gen, xt_gen, xv_gen, xte_gen))
                step_log.append(f"- Generate Strategy ('{gen_col}'): {score_gen:.4f}")

            # Candidate 3: MIXTURE
            if not inf_error and actual_dropped:
                # quick_eval on Test
                score_mix, test_mix = quick_eval(xtrain_mix, y_train, xval_mix, y_val, xtest_mix, y_test)
                candidates.append(("Mixture", score_mix, test_mix, xtrain_mix, xval_mix, xtest_mix))
                step_log.append(f"- Mixture Strategy: {score_mix:.4f}")

            # Compare candidates to Current Best (which is Test Accuracy)
            best_candidate_score = -1
            best_candidate_idx = -1
            
            for i, (name, score, test_acc, _, _, _) in enumerate(candidates):
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate_idx = i
            
            # === DECISION LOGIC ===
            
            # Case 1: Improvement Detected
            name, score, test_acc, xt, xv, xte = candidates[best_candidate_idx]
            val_test_text = f"Step {current_step}: val {score:.2f}, {test_acc:.2f}"
            logging.info(val_test_text)

            if score > global_best_val:
                global_best_val = score
                best_test_result = test_acc
                print(f"--- Global Best Updated: {global_best_val:.4f} ---")

            if score > current_best_val:
                winner_name = name
                next_xtrain, next_xval, next_xtest = xt, xv, xte
                
                improvement = score - current_best_val
                decision_text = f"ACCEPTED '{name}' strategy (Improved {current_best_val:.4f} -> {score:.4f})"
                
                # Update Best Scores
                current_best_val = score
                
                # ===== NEW: Record strategy to history with detailed actions =====
                strategy_record = {
                    "step": current_step,
                    "strategy_name": name,
                    "validation_score": score,
                    "improvement": improvement,
                    "best_score_at_step": score,
                    "description": f"Strategy '{name}' applied with score {score:.4f}",
                    "dropped_features": actual_dropped if name == "Drop Only" or name == "Mixture" else [],
                    "generated_feature": gen_col if name == "Generate Only" or name == "Mixture" else None
                }
                past_strategies_history.append(strategy_record)

            # Case 2: No Improvement -> Check Logic
            else:
                # Try to find the "Generate Only" candidate
                gen_candidate = next((c for c in candidates if c[0] == "Generate Only"), None)
                
                # Tolerance Threshold (in percentage for 0-100 scale)
                DROP_THRESHOLD = drop_threshold
                
                if gen_candidate:
                    name, score, test, xt, xv, xte = gen_candidate
                    
                    # Calculate the performance drop
                    drop = current_best_val - score
                    
                    # Sub-case 2a: Drop is too large (> 3%) -> REJECT
                    if drop > DROP_THRESHOLD:
                        decision_text = (f"REJECTED '{name}'. Performance dropped too much "
                                         f"({current_best_val:.4f} -> {score:.4f}, drop > {DROP_THRESHOLD}%).")
                        # Keep original state
                        
                    # Sub-case 2b: Drop is acceptable (<= 3%) -> EXPLORE/ACCEPT
                    else:
                        next_xtrain, next_xval, next_xtest = xt, xv, xte
                        
                        improvement = score - current_best_val  # Will be negative, but record it
                        decision_text = (f"EXPLORATION: Performance dropped ({current_best_val:.4f} -> {score:.4f}), "
                                         f"but within {DROP_THRESHOLD}% limit. Accepted '{name}' for refinement.")
                        
                        # Update baseline to this new score
                        current_best_val = score
                        
                        # ===== NEW: Record exploratory strategy with detailed actions =====
                        strategy_record = {
                            "step": current_step,
                            "strategy_name": name + " (Exploration)",
                            "validation_score": score,
                            "improvement": improvement,
                            "best_score_at_step": current_best_val,
                            "description": f"Exploratory strategy '{name}' with acceptable drop",
                            "dropped_features": [],
                            "generated_feature": gen_col if gen_col else None
                        }
                        past_strategies_history.append(strategy_record)

                else:
                    decision_text = (f"REJECTED all strategies. (Best was {best_candidate_score:.4f}, "
                                     f"and Generated feature was invalid/missing).")

            # === END DECISION LOGIC ===
            
            if current_best_val == 100.00:
                print("Perfect Accuracy Reached!")
                break

            # Construct Report for Next Step
            last_step_report = f"Previous Strategies:\n" + "\n".join(step_log) + \
                               f"\n\nDecision: {decision_text}"
            
            print(last_step_report)

            # Apply the Winner
            x_train_df, x_val_df, x_test_df = next_xtrain, next_xval, next_xtest

            current_step += 1

        end_time = time.time()
        print(f"CHAMPION SELECTION (Ablation Study) Time: {end_time-start_time}")
    return dataset_name, baseline_result, best_test_result, past_strategies_history

    
if __name__ == "__main__":
    # ===== ARGPARSE: Parse command-line arguments =====
    parser = argparse.ArgumentParser(
        description="HITL AutoFE: Human-in-the-Loop Automated Feature Engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file
  python hitl-autofe_v4.py --config config.yml
  
  # Use command-line arguments
  python hitl-autofe_v4.py --data-path /path/to/data --steps 20 --llm-model gpt-4
  
  # Mix config file and override with command-line
  python hitl-autofe_v4.py --config config.yml --steps 10 --temperature 0.5
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to config file (default: config.yml)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of optimization steps')
    parser.add_argument('--llm-model', type=str, default=None,
                        help='LLM model to use for feature engineering')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature parameter for LLM')
    parser.add_argument('--drop-threshold', type=float, default=None,
                        help='Threshold for dropping features')
    parser.add_argument('--dir-name', type=str, default=None,
                        help='Directory name for results (default: timestamp)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated list of seeds (e.g., "0,1,2,3,4")')
    parser.add_argument('--masked', type=lambda x: x.lower() in ['true', '1', 'yes'], default=None,
                        help='Whether to mask feature names (true/false)')
    parser.add_argument('--processed-datasets', type=str, default=None,
                        help='Comma-separated list of datasets to skip')
    parser.add_argument('--use-config', action='store_true', default=True,
                        help='Load from config file first (can be overridden by CLI args)')
    parser.add_argument('--port', type=int, default=None,
                        help='Local LLM Model Service Access Port')
    
    args = parser.parse_args()
    
    # ===== LOAD CONFIG FILE (if available) =====
    config = {}
    config_loaded = False
    
    if args.use_config:
        try:
            config = load_config(args.config)
            validate_config(config)
            config_loaded = True
            print(f"✓ Config loaded from {args.config}")
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            print("Using command-line arguments or defaults...")
    
    # ===== MERGE: Config file + CLI arguments =====
    # Priority: CLI args > Config file > Hardcoded defaults
    
    # data_path
    data_path = args.data_path or config.get('data_path') or "/home/xylar369/experiment_data"
    
    # steps
    steps = args.steps or config.get('steps') or 15
    
    # llm_model
    llm_model = args.llm_model or config.get('llm_model') or "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"
    
    # temperature
    temperature = args.temperature if args.temperature is not None else config.get('temperature', 0.2)
    if temperature is None:
        temperature = 0.2
    
    # drop_threshold
    drop_threshold = args.drop_threshold if args.drop_threshold is not None else config.get('drop_threshold', 1.5)
    if drop_threshold is None:
        drop_threshold = 1.5
    
    # dir_name
    dir_name = args.dir_name or config.get('dir_name')
    
    # seeds
    if args.seeds:
        seeds = args.seeds.split(',')
    else:
        seeds = config.get('seeds', ["0", "1", "2", "3", "4"])
    
    # masked
    if args.masked is not None:
        masked = args.masked
    else:
        masked = config.get('masked', True)
    
    # processed_datasets
    if args.processed_datasets:
        processed_datasets = args.processed_datasets.split(',')
    else:
        processed_datasets = config.get('processed_datasets', [])
    
    # Port
    port = args.port or config.get('port') or 8000
    
    # ===== GENERATE TIME SIGN (if dir_name not provided) =====
    if dir_name:
        time_sign = dir_name
    else:
        now = datetime.now()
        time_sign = f"{str(now.date()).replace('-','_')}_{now.hour}_{now.minute}"
    
    
    # ===== PRINT CONFIGURATION =====
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Data Path:           {data_path}")
    print(f"Steps:               {steps}")
    print(f"LLM Model:           {llm_model}")
    print(f"Temperature:         {temperature}")
    print(f"Drop Threshold:      {drop_threshold}")
    print(f"Time Sign:           {time_sign}")
    print(f"Seeds:               {seeds}")
    print(f"Masked Features:     {masked}")
    print(f"Processed Datasets:  {processed_datasets}")
    print(f"Port:  {port}")
    print("="*60 + "\n")

    llm_model_name = llm_model.split("/")[-1]
    results_dir = f"../../results/hitl_autofe/{llm_model_name}/{time_sign}/results"
    prompt_dir = f"../../results/hitl_autofe/{llm_model_name}/{time_sign}/prompt"
    config_saved_dir = f"../../results/hitl_autofe/{llm_model_name}/{time_sign}"

    summary_path =  f"../../results/hitl_autofe/{llm_model_name}/{time_sign}/summary.txt"
    summary_dict = {}

    # ===== SAVE RUNNING PARAMETERS =====
    # Create a dictionary with all resolved parameters
    running_config = {
        'data_path': data_path,
        'steps': steps,
        'llm_model': llm_model,
        'temperature': temperature,
        'drop_threshold': drop_threshold,
        'dir_name': time_sign,
        'seeds': seeds,
        'masked': masked,
        'processed_datasets': processed_datasets,
        'config_source': 'argparse' if config_loaded else 'defaults',
        'timestamp': str(datetime.now().isoformat()),
        'port': port
    }
    
    # Save the actual running configuration
    save_config(running_config, config_saved_dir, filename="config_used.yml")

    # unprocessed_datasets = ["credit-g", "diabetes", "cmc"]
    
    # struture:
    # -dataset_name:
    # --metadata.json
    # --seed_n:
    # --- train.csv
    # --- val.csv
    # --- test.csv
    for dataset_dir in os.listdir(data_path):
        if dataset_dir in processed_datasets:
            continue
        dataset_path = os.path.join(data_path, dataset_dir)
        if os.path.isdir(dataset_path):
            for seed in seeds:
                metadata_path = os.path.join(dataset_path, "metadata.json")
                seed_dir = os.path.join(dataset_path, f"seed_{seed}")
                if os.path.exists(seed_dir):
                    csv_train = os.path.join(seed_dir, "train.csv")
                    csv_val = os.path.join(seed_dir, "validation.csv")
                    csv_test = os.path.join(seed_dir, "test.csv")
                    print(f"Processing dataset: {dataset_dir} with seed {seed}")
                    dataset_name, baseline_acc, best_acc, strategies_history = processing_from_csv(
                        csv_train, csv_val, csv_test, metadata_path,
                        steps=steps, masked_flag=masked,
                        results_dir=f"{results_dir}_seed{seed}",
                        prompt_dir=f"{prompt_dir}_seed{seed}",
                        llm_model=llm_model,
                        temperature=temperature,
                        drop_threshold=drop_threshold,
                        port=port
                    )
                    summary_dict[dataset_name] = (baseline_acc, best_acc)
                    summary_text = f"{dataset_name} Seed {seed}: Baseline {baseline_acc:.2f}, Best {best_acc:.2f}, Improvement: {best_acc-baseline_acc:.2f}\n"
                    with open(summary_path, "a+") as f:
                        f.write(summary_text)
                    f.close()

        
        
        

    


    

    
