# HW_ML_1

Воспроизведение:
```bash
git clone https://github.com/nasibulinaa/hw_ml_1.git
cd hw_ml_1
pip install -r requirements.txt
dvc pull
dvc repro
```

Запуск web-интерфейса MLFlow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Основные зависимости: `dvc mlflow numpy pandas scikit-learn PyYAML`.

Генерация данных (при отсутствии storage): `python3 src/gen_data.py`

Структура:
```
├── data/ # Сырые и обработанные данные (только через DVC)
├── src/ # Скрипты (prepare.py, train.py)
├── dvc.yaml # Описание пайплайна
├── params.yaml # Гиперпараметры
├── requirements.txt # Зависимости
└── README.md # Документация 
```
