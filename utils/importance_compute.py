import pandas as pd
import shap
import numpy as np

def process_shap_importance(model, val_with_noise):
    """
    Compute SHAP importance score from model and categorize features.
    
    Parameters:
    model: The trained model (must support TreeExplainer)
    val_with_noise (pd.DataFrame): The validation set that includes a noise column.
    
    Returns:
    importance_df (pd.DataFrame): Full importance breakdown sorted by SHAP value descending.
    useful_features (list): Features with SHAP value higher than noise (excluding noise column).
    useless_features (list): Features with SHAP value lower than or equal to noise (excluding noise column).
    top_10_features (list): Top 3 features by SHAP value (excluding noise column).
    noise_threshold (float): The SHAP value of the noise column.
    """
    # Initialize with default fallback
    shap_imp = {col: 0.0 for col in val_with_noise.columns}

    try:
        explainer = shap.TreeExplainer(model)
        # Check if shap_values returns a list (multiclass) or array (binary)
        shap_values = explainer.shap_values(val_with_noise)
        
        if isinstance(shap_values, list):
            # Multiclass: take mean across classes
            shap_values = np.mean([np.abs(c) for c in shap_values], axis=0)
        else:
            # Binary/Regression
            shap_values = np.abs(shap_values)

        # Average across samples
        mean_shap = shap_values.mean(axis=0)
        
        # Ensure mean_shap is 1D
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()
        
        for i, col in enumerate(val_with_noise.columns):
            if i < len(mean_shap):
                # Extract scalar value to avoid storing arrays
                val = mean_shap[i]
                if isinstance(val, np.ndarray):
                    val = float(val.item()) if val.size == 1 else float(val.mean())
                else:
                    val = float(val)
                shap_imp[col] = val
    except Exception as e:
        print(f"SHAP Importance Failed: {e}")
        # Initialize with default fallback
        shap_imp = {col: 0.0 for col in val_with_noise.columns}

    # 1. Create DataFrame and sort by SHAP value descending
    importance_data = [{'feature': col, 'shap_value': shap_imp[col]} for col in val_with_noise.columns]
    importance_df = pd.DataFrame(importance_data).sort_values('shap_value', ascending=False).reset_index(drop=True)
    
    # 2. Identify the Noise Threshold
    # We assume the noise column is named 'noise' (or find it by searching the string)
    all_cols = val_with_noise.columns
    noise_col_names = [c for c in all_cols if 'noise' in c.lower()]
    if noise_col_names:
        noise_col_name = noise_col_names[0]
        noise_threshold = importance_df.loc[importance_df['feature'] == noise_col_name, 'shap_value'].values[0]
    else:
        # If no noise column found, use minimum value as threshold
        noise_threshold = importance_df['shap_value'].min()
    
    # 3. Categorize Features
    # Useful: SHAP value strictly greater than noise (and exclude noise itself)
    if noise_col_names:
        useful_features = importance_df[
            (importance_df['shap_value'] > noise_threshold) & 
            (importance_df['feature'] != noise_col_names[0])
        ]['feature'].tolist()
    else:
        useful_features = importance_df[importance_df['shap_value'] > noise_threshold]['feature'].tolist()
    
    # Useless: SHAP value is 0 or less than/equal to noise
    # (Excluding the noise column itself from the list if it exists)
    if noise_col_names:
        useless_features = importance_df[
            (importance_df['shap_value'] <= noise_threshold) & 
            (importance_df['feature'] != noise_col_names[0])
        ]['feature'].tolist()
    else:
        useless_features = importance_df[importance_df['shap_value'] <= noise_threshold]['feature'].tolist()
    
    # 4. Calculate "Top 3" logic - exclude noise column
    # Filter out noise column from importance_df
    if noise_col_names:
        clean_importance_df = importance_df[importance_df['feature'] != noise_col_names[0]]
    else:
        clean_importance_df = importance_df
    
    # Take top 3 features (or fewer if not enough features)
    num_top = min(3, len(clean_importance_df))
    top_10_features = clean_importance_df.head(num_top)['feature'].tolist()
    
    # 5. Remove noise column from importance_df before returning
    if noise_col_names:
        importance_df = importance_df[importance_df['feature'] != noise_col_names[0]].reset_index(drop=True)

    return importance_df, useful_features, useless_features, top_10_features, noise_threshold

def process_gain_importance(model, x_train_with_noise):
    """
    Analyzes feature importance by comparing actual features to a shadow noise column.
    
    Parameters:
    model: The trained XGBoost model (best_clf)
    x_train_with_noise (pd.DataFrame): The training set that includes the noise column.
    
    Returns:
    importance_df (pd.DataFrame): Full importance breakdown sorted by gain descending.
    useful_features (list): Features with gain higher than noise (excluding noise column).
    useless_features (list): Features with gain lower than or equal to noise (excluding noise column).
    top_10_features (list): Top 3 features by gain (excluding noise column).
    noise_threshold (float): The gain value of the noise column.
    """
    # 1. Get raw gain scores
    # XGBoost returns a dict for only the features it used
    score_dict = model.get_booster().get_score(importance_type='gain')
    
    # 2. Map scores to ALL columns (including noise and those with 0 importance)
    all_cols = x_train_with_noise.columns
    importance_data = []
    
    for col in all_cols:
        gain = score_dict.get(col, 0.0)
        importance_data.append({'feature': col, 'gain': gain})
        
    importance_df = pd.DataFrame(importance_data).sort_values('gain', ascending=False).reset_index(drop=True)
    
    # 3. Identify the Noise Threshold
    # We assume the noise column is named 'noise' (or find it by searching the string)
    noise_col_name = [c for c in all_cols if 'noise' in c.lower()][0]
    noise_threshold = importance_df.loc[importance_df['feature'] == noise_col_name, 'gain'].values[0]
    
    # 4. Categorize Features
    # Useful: Gain strictly greater than noise (and exclude noise itself)
    useful_features = importance_df[
        (importance_df['gain'] > noise_threshold) & 
        (importance_df['feature'] != noise_col_name)
    ]['feature'].tolist()
    
    # Useless: Gain is 0 or less than/equal to noise
    # (Excluding the noise column itself from the list)
    useless_features = importance_df[
        (importance_df['gain'] <= noise_threshold) & 
        (importance_df['feature'] != noise_col_name)
    ]['feature'].tolist()
    
    # 5. Calculate "Top 3" logic - exclude noise column
    # Filter out noise column from importance_df
    clean_importance_df = importance_df[importance_df['feature'] != noise_col_name]
    
    # Take top 3 features (or fewer if not enough features)
    num_top = min(3, len(clean_importance_df))
    top_10_features = clean_importance_df.head(num_top)['feature'].tolist()
    
    # 6. Remove noise column from importance_df before returning
    importance_df = importance_df[importance_df['feature'] != noise_col_name].reset_index(drop=True)

    return importance_df, useful_features, useless_features, top_10_features, noise_threshold