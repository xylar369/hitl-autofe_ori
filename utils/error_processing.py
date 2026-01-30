import numpy as np

def prcocessed_inf_column(processed_df, generated_column_name):
    """
    Check whether there are problems in the generated features
    Especially whether inf or -inf in the features
    """
    exists_inf = np.isinf(processed_df[generated_column_name]).any()
    if exists_inf:
        return processed_df.drop(columns=[generated_column_name])

    return processed_df, exists_inf
