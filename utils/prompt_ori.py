import os
import re

def get_init_prompt(task_type, dataset_name, dataset_description, val_acc, important_df, 
               useful_features, useless_features, noise_threshold, feature_mapping):
    total_features = important_df.shape[0]
    total_useful = len(useful_features)
    total_useless = len(useless_features)

    # Base Context
    prompt = f"""
### CONTEXT
I am working on a {task_type} task using the "{dataset_name}" dataset.
**Dataset Description:** {dataset_description}

### MODEL PERFORMANCE & ANALYSIS
- **Model:** XGBoost
- **Current Validation Accuracy:** {val_acc:.2f}%
- **Feature Selection Method:** Shadow-feature analysis (comparing real features against a random noise column).
- **Noise Threshold (Gain):** {noise_threshold}
- **Summary:** {total_useful} features outperformed the noise column, while {total_useless} features were statistically indistinguishable from noise.
"""

   # 1. Top Useful Features (The Heavy Hitters)
    # Since your dataset only has 5 features, 'Top 10' might be your whole dataset. 
    # Let's show the top 3 or top 10% specifically.
    top_n = max(1, int(0.1 * total_features)) if total_features > 10 else 3
    
    top_features = important_df.head(top_n)
    feature_info = "\n".join([f"- {row['feature']}: {row['gain']:.4f}" for _, row in top_features.iterrows()])
    prompt += f"\n### TOP USEFUL FEATURES (Highest Gain):\n{feature_info}\n"

    # 2. Remaining Useful Features (The "Rest")
    # These are features above the noise threshold but not in the Top section.
    useful_set = set(useful_features)
    top_set = set(top_features['feature'])
    remaining_useful = [f for f in useful_features if f not in top_set]
    
    if remaining_useful:
        remaining_df = important_df[important_df['feature'].isin(remaining_useful)]
        remaining_info = "\n".join([f"- {row['feature']}: {row['gain']:.4f}" for _, row in remaining_df.iterrows()])
        prompt += f"\n### OTHER USEFUL FEATURES (Above Noise Threshold):\n{remaining_info}\n"

    # 3. Useless Features Section
    if total_useless > 0:
        # Since you only have 5 features, we can likely show all useless ones
        useless_str = ", ".join(useless_features) 
        prompt += f"\n### USELESS FEATURES (To be dropped):\n{useless_str}\n"
    else:
        prompt += "\n### USELESS FEATURES:\nNone\n"

    # 4. Final Instructions (The "Ask")
    prompt += f"""
### CODING TASK & RESTRICTIONS
You must act as a precise Python developer. Provide a single Python code block following these strict rules:

1. **Strategy**: Your primary goal is to improve the model by dropping the identified useless features or applying **exactly one** feature engineering method (e.g., creating a new interaction feature, binning, or a mathematical transformation).
2. **No Multiple Transformations**: Do not suggest or implement multiple complex steps. Choose the single most impactful transformation.
3. **Template**: You must use the following function signature exactly:
   ```python
   def apply_feature_engineering(df):
       # Drop useless features
       # or Generate exactly one new feature or transformation
       return df, [dropped_features_names], generated_feature_name
    ```end
"""
    return prompt

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

def get_prompt(task_type, dataset_name, dataset_description, val_acc, important_df, 
               useful_features, useless_features, noise_threshold, previous_strategies, previous_val_acc,feature_mapping):
    total_features = important_df.shape[0]
    total_useful = len(useful_features)
    total_useless = len(useless_features)

    strategies_text = "\n"
    val_text = "\n"
    for strategy in previous_strategies:
        strategies_text += f"- {strategy}\n"
    for val in previous_val_acc[:-2]:
        val_text += f"- {val}%\n"

    # Base Context
    prompt = f"""
### CONTEXT
I am working on a {task_type} task using the "{dataset_name}" dataset.
**Dataset Description:** {dataset_description}
You have previously given me some strategies to improve the model.

### MODEL PERFORMANCE & ANALYSIS
- **Model:** XGBoost
- **Previous Strategies:** {strategies_text}
- **Previous Validation Accuracy:** {val_text}
- **Current Validation Accuracy:** {val_acc:.2f}%
- **Feature Selection Method:** Shadow-feature analysis (comparing real features against a random noise column).
- **Noise Threshold (Gain):** {noise_threshold}
- **Summary:** {total_useful} features outperformed the noise column, while {total_useless} features were statistically indistinguishable from noise.
"""

   # 1. Top Useful Features (The Heavy Hitters)
    # Since your dataset only has 5 features, 'Top 10' might be your whole dataset. 
    # Let's show the top 3 or top 10% specifically.
    top_n = max(1, int(0.1 * total_features)) if total_features > 10 else 3
    
    top_features = important_df.head(top_n)
    feature_info = "\n".join([f"- {row['feature']}: {row['gain']:.4f}" for _, row in top_features.iterrows()])
    prompt += f"\n### TOP USEFUL FEATURES (Highest Gain):\n{feature_info}\n"

    # 2. Remaining Useful Features (The "Rest")
    # These are features above the noise threshold but not in the Top section.
    useful_set = set(useful_features)
    top_set = set(top_features['feature'])
    remaining_useful = [f for f in useful_features if f not in top_set]
    
    if remaining_useful:
        remaining_df = important_df[important_df['feature'].isin(remaining_useful)]
        remaining_info = "\n".join([f"- {row['feature']}: {row['gain']:.4f}" for _, row in remaining_df.iterrows()])
        prompt += f"\n### OTHER USEFUL FEATURES (Above Noise Threshold):\n{remaining_info}\n"

    # 3. Useless Features Section
    if total_useless > 0:
        # Since you only have 5 features, we can likely show all useless ones
        useless_str = ", ".join(useless_features) 
        prompt += f"\n### USELESS FEATURES (To be dropped):\n{useless_str}\n"
    else:
        prompt += "\n### USELESS FEATURES:\nNone\n"

    # 4. Final Instructions (The "Ask")
    prompt += f"""
### CODING TASK & RESTRICTIONS
You must act as a precise Python developer. 

1. **Reflection**: Review the previously suggested strategies and their impact on validation accuracy. Consider whether dropping certain features or generating a new feature could further enhance model performance.
2. **Strategy**: Your primary goal is to improve the model by dropping some features (identified useless or generated) or applying **exactly one** feature engineering method (e.g., creating a new interaction feature, binning, or a mathematical transformation). 
3. **Thinkings**: Be Creative but Pragmatic. Consider feature interactions, polynomial features, or domain-specific transformations that could unveil hidden patterns in the data.
4. **Template**: You must use the following function signature exactly:
   ```python
   def apply_feature_engineering(df):
       # Drop useless features
       # or Generate exactly one new feature or transformation
       return df, [dropped_features_names], generated_feature_name
    ```end
"""
    return prompt

# 2. **No Multiple Transformations**: Do not suggest or implement multiple complex steps. Choose the single most impactful transformation.?

