import pandas as pd
import numpy as np
import traceback
import importlib.util

def run_feature_engineering(xtrain, xval, xtest, previous_strategies, step, code_file="running_code.py", log_file="errors.log"):
    """
    Dynamically loads the generated code and applies it to the datasets.
    If it fails, logs the error and returns original data.
    """
    try:
        # 1. Dynamically import the generated script
        spec = importlib.util.spec_from_file_location("dynamic_feature_eng", code_file)
        module = importlib.util.module_from_spec(spec)

        module.np = np
        module.pd = pd
        
        spec.loader.exec_module(module)

        
        # 2. Check if the function exists
        if hasattr(module, 'apply_feature_engineering'):
            print("Applying generated feature engineering...")
            # Apply to all splits
            xtrain, drop_features, generate_feature = module.apply_feature_engineering(xtrain)
            xval, _, _ = module.apply_feature_engineering(xval)
            xtest, _, _ = module.apply_feature_engineering(xtest)
            print("Successfully updated datasets.")
            if drop_features:
                text = f"Step {step}: Dropped "
                for dropped in drop_features:
                    text += f"'{dropped}', "
                previous_strategies.append(text)
            if generate_feature:
                text = f"Step {step}: Generated feature '{generate_feature}'"
                previous_strategies.append(text)
        else:
            raise AttributeError("Function 'apply_feature_engineering' not found in script.")

    except Exception as e:
        # 3. Log errors to file instead of stopping execution
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"--- Error with generated code in {code_file} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        print(f"Error encountered. Check {log_file} for details. Proceeding with original data.")

    return xtrain, xval, xtest

def run_feature_engineering_traced(xtrain, xval, xtest, previous_strategies, step, code_file="running_code.py", log_file="errors.log"):
    """
    Dynamically loads the generated code and applies it to the datasets.
    If it fails, logs the error and returns original data.
    """
    xtrain_ori, xval_ori, xtest_ori = xtrain, xval, xtest
    xtrain_dropped, xval_dropped, xtest_dropped = None, None, None
    xtrain_generated, xval_generated, xtest_generated = None, None, None
    xtrain_mixed, xval_mixed, xtest_mixed = None, None, None

    try:
        # 1. Dynamically import the generated script
        spec = importlib.util.spec_from_file_location("dynamic_feature_eng", code_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 2. Check if the function exists
        if hasattr(module, 'apply_feature_engineering'):
            print("Applying generated feature engineering...")
            # Apply to all splits
            xtrain_mixed, drop_features, generate_feature = module.apply_feature_engineering(xtrain)
            xval_mixed, _, _ = module.apply_feature_engineering(xval)
            xtest_mixed, _, _ = module.apply_feature_engineering(xtest)
            print("Successfully updated datasets.")
            if drop_features:
                text = f"Step {step}: Dropped "
                for dropped in drop_features:
                    text += f"'{dropped}', "
                previous_strategies.append(text)

            if generate_feature:
                text = f"Step {step}: Generated feature '{generate_feature}'"
                previous_strategies.append(text)
        else:
            raise AttributeError("Function 'apply_feature_engineering' not found in script.")

    except Exception as e:
        # 3. Log errors to file instead of stopping execution
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"--- Error with generated code in {code_file} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
        print(f"Error encountered. Check {log_file} for details. Proceeding with original data.")

    return xtrain, xval, xtest

# --- Helper: Apply Code & Extract Intent ---
def safe_apply_feature_engineering(code_file, xtrain, xval, xtest):
    """
    Loads the code, runs it, and returns the modified datasets AND the metadata
    (dropped_cols, generated_col_name).
    """
    try:
        spec = importlib.util.spec_from_file_location("dynamic_feature_eng", code_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Apply to all
        xtrain_new, dropped, generated = module.apply_feature_engineering(xtrain.copy())
        xval_new, _, _ = module.apply_feature_engineering(xval.copy())
        xtest_new, _, _ = module.apply_feature_engineering(xtest.copy())
        
        return xtrain_new, xval_new, xtest_new, dropped, generated, None
    except Exception as e:
        return None, None, None, [], None, str(e)
