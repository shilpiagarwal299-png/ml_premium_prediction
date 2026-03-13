import pandas as pd
import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,'artifacts/model_data.joblib')

model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
cols_to_scale = model_data['cols_to_scale']
features = model_data['features']

print(cols_to_scale)
print(features)

