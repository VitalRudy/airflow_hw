import os
import pandas as pd
import dill  # Используем dill, как и в pipeline
import json

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'model.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'predictions', 'predictions.csv')


def load_model(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def load_test_data(directory):
    dfs = []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Test directory not found: {directory}")

    for file in os.listdir(directory):
        if file.endswith('.json'):
            with open(os.path.join(directory, file), 'r') as f:
                data = json.load(f)
                df = pd.DataFrame([data])  # одна строка из одного JSON-объекта
                dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    return pd.concat(dfs, ignore_index=True)


def make_predictions(model, data):
    return model.predict(data)


def save_predictions(preds):
    df = pd.DataFrame(preds, columns=['prediction'])
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    df.to_csv(PREDICTIONS_PATH, index=False)


def predict():
    model = load_model(MODEL_PATH)
    test_data = load_test_data(TEST_DATA_PATH)
    preds = make_predictions(model, test_data)
    save_predictions(preds)


if __name__ == '__main__':
    predict()







