from utils.data import download_and_save_dataset, train_test_dataset_build, load_dataset_from_csv
from utils.prompt import get_prompt, extract_and_save_code
from utils.xgboost_train import train_best_xgboost_model
from utils.run_code import run_feature_engineering
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime


now = datetime.now()
print(str(now.date()).replace("-","_"))
print(now.hour)
print(now.minute)


