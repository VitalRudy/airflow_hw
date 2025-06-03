import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Абсолютный путь к airflow_hw, чтобы импортировать pipeline и predict
sys.path.append('/Users/apple/PycharmProjects/PythonProject/airflow_hw')

from modules.pipeline import pipeline
from modules.predict import predict

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022, 6, 1),
    'retries': 1,
}

with DAG(
    dag_id='car_price_prediction',
    default_args=default_args,
    schedule='15 0 * * *',  # каждый день в 00:15
    catchup=False,
    tags=['ml'],
) as dag:

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=pipeline,
    )

    predict_task = PythonOperator(
        task_id='predict_prices',
        python_callable=predict,
    )

    train_task >> predict_task






