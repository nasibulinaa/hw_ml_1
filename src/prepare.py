import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]
raw_data_path = "data/raw/data.csv"
target_data_dir = "data/processed/"
os.makedirs(target_data_dir, exist_ok=True)

# Now split data to train and test
data = pd.read_csv(raw_data_path)
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=42)

data_train = X_train
data_train['target']=y_train
data_test = X_test
data_test['target']=y_test

data_train.to_csv(f"{target_data_dir}/data_train.csv", index=False)
data_test.to_csv(f"{target_data_dir}/data_test.csv", index=False)
