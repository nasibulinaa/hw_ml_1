import pandas as pd
import os
from sklearn.datasets import load_iris

DATA_DIR="data"
DATA_RAW_DIR=os.path.join(DATA_DIR, "raw")
IRIS_PATH=os.path.join(DATA_RAW_DIR, "data.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_RAW_DIR, exist_ok=True)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target']=iris.target
data.to_csv(IRIS_PATH, index=False)