# downloading dataset from openml repository
from openml import datasets
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def download_and_save_dataset(dataset_id, save_dir="data/", database="openml"):
    """
    Load a dataset from OpenML by its ID and save it to a specified path.

    Parameters:
    dataset_id (int): The OpenML dataset ID.
    save_path (str): The file path where the dataset will be saved. default is "data/".
    database (str): The database to load the dataset from. Default is "openml".

    Returns:
    None
    """
    if database == "openml":
        # Load the dataset from OpenML
        dataset = datasets.get_dataset(dataset_id)
        
        # Get the data as a pandas DataFrame
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')
        
        # Combine features and target into a single DataFrame
        data = X.copy()
        data[dataset.default_target_attribute] = y

        # get dataset name and create save path
        dataset_name = dataset.name.replace(" ", "_")
        save_path = os.path.join(save_dir, f"{dataset_name}.csv")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the DataFrame to a CSV file
        data.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
        # Save the dataset description
        with open(save_path.replace(".csv", "_description.txt"), "w") as f:
            f.write(dataset.description)

def minmax_scale_datasets(x_train, x_val, x_test):
    """
    Apply Min-Max Scaling to training, validation, and test datasets.

    Parameters:
    x_train (pd.DataFrame): Training features.
    x_val (pd.DataFrame): Validation features.
    x_test (pd.DataFrame): Test features.

    Returns:
    pd.DataFrame, pd.DataFrame, pd.DataFrame: Scaled training, validation, and test features.
    """
    scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    return x_train_scaled, x_val_scaled, x_test_scaled

def train_test_dataset_build(data, test_size=0.2, val_size=0.1):
    """
    Processes the dataframe using Label Encoding for categorical features.
    Splits the data into training, validation, and test sets.

    returns:
    xtrain, xval, xtest, ytrain, yval, ytest,[df]
    """
    # 1. Clean data
    data = data.dropna().copy()

    # 2. Separate Features (X) and Target (y)
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()

    # 3. Identify Original Columns and Categorical Columns
    original_cols = X.columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Apply Label Encoding
    # We transform strings into numbers (e.g., "AA" -> 0, "AS" -> 1)
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    y_encoder = None
    # If target is categorical object or pandas Categorical, use LabelEncoder on strings
    if y.dtype == 'object' or hasattr(y, 'cat'):
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))
    else:
        # If the target is numeric (int/float) but encodes classes like [1,2,3]
        # we want to map them to 0..n_classes-1 so downstream code expecting
        # zero-based class indices works correctly. We only relabel when the
        # unique numeric labels are not already 0-based contiguous integers.
        try:
            uniq = np.unique(y)
            # Check if unique values are exactly [0, 1, ..., n-1]
            if not (uniq.size > 0 and np.array_equal(uniq, np.arange(0, uniq.size))):
                # Only apply relabeling for discrete sets (classification-like)
                # Avoid touching continuous regression targets by checking the
                # number of unique values is reasonably small relative to length.
                if uniq.size < 0.5 * len(y) or np.issubdtype(uniq.dtype, np.integer):
                    y_encoder = LabelEncoder()
                    y = y_encoder.fit_transform(y)
        except Exception:
            # If anything unexpected happens, leave y as-is and keep y_encoder None
            y_encoder = None


    # 5. Create Mapping (Simple 1-to-1 for Label Encoding)
    # Since we didn't split columns, the mapping is straightforward
    feature_mapping = {col: [col] for col in original_cols}

    # 6. Split: Train, Validation, and Test
    x_temp, xtest, y_temp, ytest = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    relative_val_size = val_size / (1 - test_size)
    xtrain, xval, ytrain, yval = train_test_split(
        x_temp, y_temp, test_size=relative_val_size, random_state=42
    )

    # 7. Min-Max Scaling
    # We scale everything to [0, 1]. This includes the new label-encoded integers.
    # scaler = MinMaxScaler()
    # xtrain_scaled = scaler.fit_transform(xtrain)
    # xval_scaled = scaler.transform(xval)
    # xtest_scaled = scaler.transform(xtest)
    
    # # Reconstruct DataFrames
    # xtrain = pd.DataFrame(xtrain_scaled, columns=original_cols)
    # xval = pd.DataFrame(xval_scaled, columns=original_cols)
    # xtest = pd.DataFrame(xtest_scaled, columns=original_cols)

    return xtrain, xval, xtest, ytrain, yval, ytest, feature_mapping

def get_data_domains(df):
    # Filter for numeric columns only to avoid errors with text data
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Efficiently calculate min and max in one go
    stats = numeric_df.agg(['min', 'max'])
    
    # Convert to the desired dictionary format: {feature: [min, max]}
    domain_dict = {}
    for col in numeric_df.columns:
        domain_dict[col] = [stats.loc['min', col], stats.loc['max', col]]
        
    return domain_dict

import pandas as pd
import numpy as np

def get_columns_with_inf(df):
    """
    Checks the DataFrame for infinite values and returns a list of column names containing them.
    
    Args:
        df (pd.DataFrame): The dataframe to check.
        
    Returns:
        list: A list of column names that contain infinite values.
    """
    # select_dtypes(include=[np.number]) ensures we only check numeric columns
    # to avoid errors on string/object columns.
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Check for inf or -inf
    is_inf = np.isinf(numeric_df)
    
    # Get columns where any value is True
    inf_cols = numeric_df.columns[is_inf.any()].tolist()
    
    return inf_cols

def csv_processing(data):
    """
    Processes the dataframe using Label Encoding for categorical features.
    Splits the data into training, validation, and test sets.

    returns:
    xtrain, xval, xtest, ytrain, yval, ytest,[df]
    """
    # 1. Clean data
    data = data.dropna().copy()

    # 2. Separate Features (X) and Target (y)
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()

    # 3. Identify Original Columns and Categorical Columns
    original_cols = X.columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Apply Label Encoding
    # We transform strings into numbers (e.g., "AA" -> 0, "AS" -> 1)
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    y_encoder = None
    # If target is categorical object or pandas Categorical, use LabelEncoder on strings
    if y.dtype == 'object' or hasattr(y, 'cat'):
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))
    else:
        # If the target is numeric (int/float) but encodes classes like [1,2,3]
        # we want to map them to 0..n_classes-1 so downstream code expecting
        # zero-based class indices works correctly. We only relabel when the
        # unique numeric labels are not already 0-based contiguous integers.
        try:
            uniq = np.unique(y)
            # Check if unique values are exactly [0, 1, ..., n-1]
            if not (uniq.size > 0 and np.array_equal(uniq, np.arange(0, uniq.size))):
                # Only apply relabeling for discrete sets (classification-like)
                # Avoid touching continuous regression targets by checking the
                # number of unique values is reasonably small relative to length.
                if uniq.size < 0.5 * len(y) or np.issubdtype(uniq.dtype, np.integer):
                    y_encoder = LabelEncoder()
                    y = y_encoder.fit_transform(y)
        except Exception:
            # If anything unexpected happens, leave y as-is and keep y_encoder None
            y_encoder = None


    return X, y

def load_dataset_from_csv(file_path):
    """
    Load a dataset from a CSV file.

    Parameters:
    file_path (str): The file path of the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def mask_feature_name(df):
    num_features  = len(df.columns) - 1

    new_names = [f"f{i+1}" for i in range(num_features)] + [df.columns[-1]]

    df.columns = new_names

    return df

def get_X_y(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y
    
