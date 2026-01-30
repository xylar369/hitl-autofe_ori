import os
import re

def generate_feature_prompt(domain_dict):
    """
    Converts the domain dictionary into a formatted string.
    Format: feature name [min, max]; ...
    """
    prompt_parts = []
    for feature, (min_val, max_val) in domain_dict.items():
        # Formatting floats to 2 decimal places for cleaner text, 
        # remove ':.2f' if you want full precision.
        part = f"{feature} [{min_val:.2f}, {max_val:.2f}]"
        prompt_parts.append(part)
    
    # Join all parts with a semicolon and space
    return "; ".join(prompt_parts)

def extract_and_save_code(llm_response, output_filename="running_code.py"):
    """
    Extracts Python code from a Markdown response and saves it to a file.
    """
    # Regex to find content between ```python and ```
    # The 're.DOTALL' flag ensures that '.' matches newlines as well
    code_pattern = r"```python\n(.*?)\n```"
    match = re.search(code_pattern, llm_response, re.DOTALL)
    
    if match:
        code_content = match.group(1)
        
        # Save to file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(code_content)
        
        print(f"Successfully extracted code to {output_filename}")
        return True
    else:
        print("No Python code block found in the response.")
        return False

def get_prompt_v4(task_type, dataset_name, dataset_description, val_acc, important_df, masked_flag,
                           useful_features, useless_features, noise_threshold, 
                           last_step_report, domain_dict, past_strategies_history=None):
    """
    Similar to get_prompt_v3, but includes the best past optimal strategy.
    
    Parameters:
    - past_strategies_history: List of strategy dicts with keys: step, strategy_name, 
      validation_score, improvement, best_score_at_step, description
      Only the BEST strategy (highest improvement) is shown.
    """
    
    # --- Feature Statistics ---
    total_features = important_df.shape[0]
    top_n = max(3, int(0.1 * total_features)) if total_features > 10 else 5
    top_features_df = important_df.head(top_n)
    top_feature_names = top_features_df["feature"].tolist()
    
    # --- Identify "Other Useful Features" ---
    other_useful_features = [f for f in useful_features if f not in top_feature_names]

    # String formatting for feature values (min/max)
    feature_vals = generate_feature_prompt(domain_dict) 
    
    prompt_head = f"""
### CONTEXT
I am working on a {task_type} task.
"""
    if not masked_flag:
        prompt_head += f"**Dataset Name**: {dataset_name}\n"
        prompt_head += f"**Dataset Description:** {dataset_description}\n"
    
    prompt_head += f"**Feature Values (Min/Max)**: {feature_vals}\n"

    # --- Build Best Past Strategy Section (WITH DETAILED ACTIONS) ---
    best_strategy_section = ""
    if past_strategies_history and len(past_strategies_history) > 1:
        # Skip baseline (index 0) and find the best strategy
        accepted_strategies = past_strategies_history[1:]
        
        if accepted_strategies:
            # Find the strategy with the highest improvement
            best_strat = max(accepted_strategies, key=lambda x: x['validation_score'])
            
            step = best_strat.get('step', 'N/A')
            name = best_strat.get('strategy_name', 'Unknown')
            score = best_strat.get('validation_score', 0.0)
            improvement = best_strat.get('improvement', 0.0)
            dropped_features = best_strat.get('dropped_features', [])
            generated_feature = best_strat.get('generated_feature', None)
            
            # Format improvement with arrow
            if improvement > 0:
                imp_str = f"{improvement:.2f}%"
            elif improvement < 0:
                imp_str = f"{improvement:.2f}%"
            else:
                imp_str = "0.00%"
            
            # Add best strategy info with detailed actions
            best_strategy_section = f"\n**Best Strategy So Far**: {name} at Step {step}\n"
            best_strategy_section += f"- **Score**: {score:.2f}%\n"
            best_strategy_section += f"- **Improvement**: {imp_str}\n"
            
            # Add detailed actions (what was done)
            if dropped_features or generated_feature:
                best_strategy_section += f"- **Actions**:\n"
                if dropped_features:
                    dropped_str = ", ".join(dropped_features)
                    best_strategy_section += f"  1. Drop {dropped_str}\n"
                if generated_feature:
                    best_strategy_section += f"  2. Generate {generated_feature}\n"

    # --- Constructing the Performance & Feedback Section ---
    prompt = f"""{prompt_head}

### CURRENT STATUS & FEEDBACK
We are iterating to improve the model. Here is the result of the **previous step**:

{last_step_report}

**Current Baseline Validation Accuracy:** {val_acc:.2f}%
(Note: The dataframe has been updated to the state of the "Accepted Strategy" above. If "Rejected", we are back to the previous best state.){best_strategy_section}

### FEATURE ANALYSIS (on Current Data)
- **Selection Method:** Shadow-feature analysis (Real vs Random Noise).
- **Noise Threshold (Gain):** {noise_threshold:.4f}
"""

    # 1. Top Features (Detailed Gain)
    feature_info = "\n".join([f"- {row['feature']}: {row['gain']:.4f}" for _, row in top_features_df.iterrows()])
    prompt += f"\n**Top Useful Features (Highest Gain):**\n{feature_info}\n"

    # 2. Other Useful Features
    if other_useful_features:
        gain_lookup = dict(zip(important_df['feature'], important_df['gain']))
        other_info_list = []
        for feat in other_useful_features:
            gain_val = gain_lookup.get(feat, 0.0)
            other_info_list.append(f"{feat} ({gain_val:.4f})")
        
        other_info_str = ", ".join(other_info_list)
        prompt += f"\n**Other Useful Features (Above Threshold):** {other_info_str}\n"
    else:
        prompt += "\n**Other Useful Features:** None.\n"

    # 3. Useless Features
    if useless_features:
        useless_str = ", ".join(useless_features)
        prompt += f"\n**Useless Features (candidates for dropping):** {useless_str}\n"
    else:
        prompt += "\n**Useless Features:** None found.\n"

    # 4. Final Instructions
    prompt += f"""
### CODING TASK
You must act as a precise Python developer.

1. **Analyze**: Look at the "Last Step Report" and the "Best Strategy So Far".
   - If a previous strategy failed or was rejected, try a different approach.
   - Learn from the best strategy but try to improve further.
   - Consider what made the best strategy effective.

2. **Strategy**: 
   - **Drop**: You may drop features listed as "Useless".
   - **Generate**: Create **exactly one** new powerful feature.
   - **Safety**: Handle division carefully (e.g., `df['a'] / (df['b'] + 1e-5)`).

3. **Output**: Write a function that returns the modified dataframe, list of dropped columns, and new feature name.

```python
def apply_feature_engineering(df):
    # 1. Drop useless features
    dropped = ['feature_name']
    df = df.drop(columns=dropped, errors='ignore')

    # 2. Generate ONE new feature
    new_col = 'f1_div_f2'
    df[new_col] = df['f1'] / (df['f2'] + 1e-6)
    
    return df, dropped, new_col
```end
"""
    return prompt

def get_prompt_v4_clean(task_type, dataset_name, dataset_description, val_acc, important_df, masked_flag,
                           useful_features, useless_features, noise_threshold, 
                           last_step_report, domain_dict, past_strategies_history=None):
    """
    Similar to get_prompt_v3, but includes the best past optimal strategy.
    
    Parameters:
    - past_strategies_history: List of strategy dicts with keys: step, strategy_name, 
      validation_score, improvement, best_score_at_step, description
      Only the BEST strategy (highest improvement) is shown.
    """
    
    # --- Feature Statistics ---
    total_features = important_df.shape[0]
    top_n = max(3, int(0.1 * total_features)) if total_features > 10 else 5
    top_features_df = important_df.head(top_n)
    top_feature_names = top_features_df["feature"].tolist()
    
    # --- Identify "Other Useful Features" ---
    other_useful_features = [f for f in useful_features if f not in top_feature_names]

    # String formatting for feature values (min/max)
    feature_vals = generate_feature_prompt(domain_dict) 
    
    prompt_head = f"""
I am working on a {task_type} task.
"""
    if not masked_flag:
        prompt_head += f"Dataset Name: {dataset_name}\n"
        prompt_head += f"Dataset Description: {dataset_description}\n"
    
    prompt_head += f"Feature Values (Min/Max):\n{feature_vals}\n"

    # --- Build Best Past Strategy Section (WITH DETAILED ACTIONS) ---
    best_strategy_section = ""
    if past_strategies_history and len(past_strategies_history) > 1:
        # Skip baseline (index 0) and find the best strategy
        accepted_strategies = past_strategies_history[1:]
        
        if accepted_strategies:
            # Find the strategy with the highest improvement
            best_strat = max(accepted_strategies, key=lambda x: x['validation_score'])
            
            step = best_strat.get('step', 'N/A')
            name = best_strat.get('strategy_name', 'Unknown')
            score = best_strat.get('validation_score', 0.0)
            improvement = best_strat.get('improvement', 0.0)
            dropped_features = best_strat.get('dropped_features', [])
            generated_feature = best_strat.get('generated_feature', None)
            
            # Format improvement with arrow
            if improvement > 0:
                imp_str = f"{improvement:.2f}%"
            elif improvement < 0:
                imp_str = f"{improvement:.2f}%"
            else:
                imp_str = "0.00%"
            
            # Add best strategy info with detailed actions
            best_strategy_section = f"\nBest Strategy So Far: {name} at Step {step}\n"
            best_strategy_section += f"- Score: {score:.2f}%\n"
            best_strategy_section += f"- Improvement: {imp_str}\n"
            
            # Add detailed actions (what was done)
            if dropped_features or generated_feature:
                best_strategy_section += f"- Actions:\n"
                if dropped_features:
                    dropped_str = ", ".join(dropped_features)
                    best_strategy_section += f"  1. Drop {dropped_str}\n"
                if generated_feature:
                    best_strategy_section += f"  2. Generate {generated_feature}\n"

    # --- Constructing the Performance & Feedback Section ---
    prompt = f"""{prompt_head}

CURRENT STATUS & FEEDBACK
We are iterating to improve the model. Here is the result of the previous step:

{last_step_report}

Current Baseline Validation Accuracy: {val_acc:.2f}%
(Note: The dataframe has been updated to the state of the "Accepted Strategy" above. If "Rejected", we are back to the previous best state.){best_strategy_section}

FEATURE ANALYSIS (on Current Data)
- Selection Method: Shadow-feature analysis (Real vs Random Noise).
- Noise Threshold (Gain): {noise_threshold:.4f}
"""

    # 1. Top Features (Detailed Gain)
    feature_info = "\n".join([f"- {row['feature']}: {row['gain']:.4f}" for _, row in top_features_df.iterrows()])
    prompt += f"\nTop Useful Features (Highest Gain):\n{feature_info}\n"

    # 2. Other Useful Features
    if other_useful_features:
        gain_lookup = dict(zip(important_df['feature'], important_df['gain']))
        other_info_list = []
        for feat in other_useful_features:
            gain_val = gain_lookup.get(feat, 0.0)
            other_info_list.append(f"{feat} ({gain_val:.4f})")
        
        other_info_str = ", ".join(other_info_list)
        prompt += f"\nOther Useful Features (Above Threshold): {other_info_str}\n"
    else:
        prompt += "\nOther Useful Features:** None.\n"

    # 3. Useless Features
    if useless_features:
        useless_str = ", ".join(useless_features)
        prompt += f"\nUseless Features (candidates for dropping): {useless_str}\n"
    else:
        prompt += "\nUseless Features: None found.\n"

    # 4. Final Instructions
    prompt += f"""
CODING TASK
You must act as a precise Python developer and data analysis expert.

1. Analyze: Look at the "Last Step Report" and the "Best Strategy So Far".
   - If a previous strategy failed or was rejected, try a different approach.
   - Learn from the best strategy but try to improve further.
   - Consider what made the best strategy effective.

2. Strategy: 
   - Drop: You may drop features listed as "Useless".
   - Generate: Create exactly one new powerful feature.
   - Safety: Handle division carefully (e.g., `df['a'] / (df['b'] + 1e-5)`).

3. Output: Write a function that returns the modified dataframe, list of dropped columns, and new feature name.

```python
def apply_feature_engineering(df):
    # 1. Drop useless features
    dropped = ['feature_name']
    df = df.drop(columns=dropped, errors='ignore')

    # 2. Generate ONE new feature
    new_col = 'f1_div_f2'
    df[new_col] = df['f1'] / (df['f2'] + 1e-6)
    
    return df, dropped, new_col
```end
"""
    return prompt

def get_prompt_shap(task_type, dataset_name, dataset_description, val_acc, important_df, masked_flag,
                           useful_features, useless_features, noise_threshold, 
                           last_step_report, domain_dict, past_strategies_history=None):
    """
    Similar to get_prompt_v3, but includes the best past optimal strategy.
    
    Parameters:
    - past_strategies_history: List of strategy dicts with keys: step, strategy_name, 
      validation_score, improvement, best_score_at_step, description
      Only the BEST strategy (highest improvement) is shown.
    """
    
    # --- Feature Statistics ---
    total_features = important_df.shape[0]
    
    # Handle empty DataFrame
    if total_features == 0:
        top_feature_names = []
        other_useful_features = useful_features
    else:
        top_n = max(3, int(0.1 * total_features)) if total_features > 10 else 5
        top_features_df = important_df.head(top_n)
        top_feature_names = top_features_df["feature"].tolist()
        
        # --- Identify "Other Useful Features" ---
        other_useful_features = [f for f in useful_features if f not in top_feature_names]

    # String formatting for feature values (min/max)
    feature_vals = generate_feature_prompt(domain_dict) 
    
    prompt_head = f"""
I am working on a {task_type} task.
"""
    if not masked_flag:
        prompt_head += f"Dataset Name: {dataset_name}\n"
        prompt_head += f"Dataset Description: {dataset_description}\n"
    
    prompt_head += f"Feature Values (Min/Max):\n{feature_vals}\n"

    # --- Build Best Past Strategy Section (WITH DETAILED ACTIONS) ---
    best_strategy_section = ""
    if past_strategies_history and len(past_strategies_history) > 1:
        # Skip baseline (index 0) and find the best strategy
        accepted_strategies = past_strategies_history[1:]
        
        if accepted_strategies:
            # Find the strategy with the highest improvement
            best_strat = max(accepted_strategies, key=lambda x: x['validation_score'])
            
            step = best_strat.get('step', 'N/A')
            name = best_strat.get('strategy_name', 'Unknown')
            score = best_strat.get('validation_score', 0.0)
            improvement = best_strat.get('improvement', 0.0)
            dropped_features = best_strat.get('dropped_features', [])
            generated_feature = best_strat.get('generated_feature', None)
            
            # Format improvement with arrow
            if improvement > 0:
                imp_str = f"{improvement:.2f}%"
            elif improvement < 0:
                imp_str = f"{improvement:.2f}%"
            else:
                imp_str = "0.00%"
            
            # Add best strategy info with detailed actions
            best_strategy_section = f"\nBest Strategy So Far: {name} at Step {step}\n"
            best_strategy_section += f"- Score: {score:.2f}%\n"
            best_strategy_section += f"- Improvement: {imp_str}\n"
            
            # Add detailed actions (what was done)
            if dropped_features or generated_feature:
                best_strategy_section += f"- Actions:\n"
                if dropped_features:
                    dropped_str = ", ".join(dropped_features)
                    best_strategy_section += f"  1. Drop {dropped_str}\n"
                if generated_feature:
                    best_strategy_section += f"  2. Generate {generated_feature}\n"

    # --- Constructing the Performance & Feedback Section ---
    prompt = f"""{prompt_head}

CURRENT STATUS & FEEDBACK
We are iterating to improve the model. Here is the result of the previous step:

{last_step_report}

Current Baseline Validation Accuracy: {val_acc:.2f}%
(Note: The dataframe has been updated to the state of the "Accepted Strategy" above. If "Rejected", we are back to the previous best state.){best_strategy_section}

FEATURE ANALYSIS (on Current Data)
- Selection Method: Shadow-feature analysis (Real vs Random Noise).
- Noise Threshold (SHAP): {noise_threshold:.4f}
"""

    # 1. Top Features (Detailed SHAP)
    feature_info = "\n".join([f"- {row['feature']}: {row['shap_value']:.4f}" for _, row in top_features_df.iterrows()])
    prompt += f"\nTop Useful Features (Highest SHAP):\n{feature_info}\n"

    # 2. Other Useful Features
    if other_useful_features:
        shap_lookup = dict(zip(important_df['feature'], important_df['shap_value']))
        other_info_list = []
        for feat in other_useful_features:
            shap_val = shap_lookup.get(feat, 0.0)
            other_info_list.append(f"{feat} ({shap_val:.4f})")
        
        other_info_str = ", ".join(other_info_list)
        prompt += f"\nOther Useful Features (Above Threshold): {other_info_str}\n"
    else:
        prompt += "\nOther Useful Features:** None.\n"

    # 3. Useless Features
    if useless_features:
        useless_str = ", ".join(useless_features)
        prompt += f"\nUseless Features (candidates for dropping): {useless_str}\n"
    else:
        prompt += "\nUseless Features: None found.\n"

    # 4. Final Instructions
    prompt += f"""
CODING TASK
You must act as a precise Python developer and data analysis expert.

1. Analyze: Look at the "Last Step Report" and the "Best Strategy So Far".
   - If a previous strategy failed or was rejected, try a different approach.
   - Learn from the best strategy but try to improve further.
   - Consider what made the best strategy effective.

2. Strategy: 
   - Drop: You may drop features listed as "Useless".
   - Generate: Create exactly one new powerful feature.
   - Safety: Handle division carefully (e.g., `df['a'] / (df['b'] + 1e-5)`).

3. Output: Write a function that returns the modified dataframe, list of dropped columns, and new feature name.

```python
def apply_feature_engineering(df):
    # 1. Drop useless features
    dropped = ['feature_name']
    df = df.drop(columns=dropped, errors='ignore')

    # 2. Generate ONE new feature
    new_col = 'f1_div_f2'
    df[new_col] = df['f1'] / (df['f2'] + 1e-6)
    
    return df, dropped, new_col
```end
"""
    return prompt