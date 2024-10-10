from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.empty import EmptyOperator
import pandas as pd
from datetime import datetime
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


# DAG Definition
@dag(schedule_interval="0 7 * * *", start_date=datetime(2024, 9, 1), catchup=False)
def randa_dag_etl():
    start_task = EmptyOperator(task_id="start_task")
    end_task = EmptyOperator(task_id="end_task")

    # Extract Task
    @task
    def extract_from_mysql():
        mysql_hook = MySqlHook("mysql_dibimbing").get_sqlalchemy_engine()

        with mysql_hook.connect() as conn:
            df = pd.read_sql(sql="SELECT * FROM dibimbing.users", con=conn)
            df['extracted_at'] = datetime.utcnow()
            #df.head()

        # Save extracted data to Parquet format in staging area
        df.to_parquet("data/users.parquet", index=False)
        print("Data successfully extracted and stored as Parquet")
        return "data/users.parquet"

    # Load Task
    @task
    def load_to_postgres(parquet_file):
        # Read from Parquet file
        df = pd.read_parquet(parquet_file)

        # Establish connection to Postgres
        postgres_hook = PostgresHook("postgres_dibimbing").get_sqlalchemy_engine()

        with postgres_hook.connect() as conn:
            df.to_sql("users", con=conn, if_exists="append", index=False)
        print("Data successfully loaded to Postgres")
    trigger_aggregation = TriggerDagRunOperator(
        task_id="trigger_aggregation",
        trigger_dag_id="randa_dag_aggregate",  # ID of the aggregation DAG
        wait_for_completion=True,  # Wait until the triggered DAG completes
        poke_interval=5,  # Check for completion every 5 seconds
    )
    # Define Task Dependencies
    extracted_data = extract_from_mysql()
    load_to_postgres_task = load_to_postgres(extracted_data)

    start_task >> extracted_data >> load_to_postgres_task >> trigger_aggregation >> end_task


etl_dag = randa_dag_etl()
