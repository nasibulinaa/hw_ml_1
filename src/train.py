import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml
import pickle
from sklearn.metrics import mean_squared_error, accuracy_score

# Загружаем параметры
params = yaml.safe_load(open("params.yaml"))["train"]

# DVC читает данные отслеживаемые в data.csv.dvc
df_train = pd.read_csv("data/processed/data_train.csv")
X_train = df_train.drop(['target'], axis=1)
y_train = df_train["target"]
df_test = pd.read_csv("data/processed/data_test.csv")
X_test = df_test.drop(['target'], axis=1)
y_test = df_test["target"]

model = RandomForestClassifier(random_state=params["random_state"], n_estimators=params["n_estimators"])
model.fit(X_train, y_train)
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Логируем в MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("dvc_pipeline_experiment")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", params["n_estimators"])
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("accuracy", accuracy)
    #mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("model.pkl")

# Сохраняем метрики для DVC
with open("metrics.json", "w") as f:
    f.write('{"mse": %f, "mse": %f}' % (mse, accuracy))
