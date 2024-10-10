from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import pandas as pd

# DAG Definition
@dag(schedule_interval="0 8 * * *", start_date=datetime(2024, 9, 1), catchup=False)
def randa_dag_aggregate():
    start_task = EmptyOperator(task_id="start_task")
    end_task = EmptyOperator(task_id="end_task")

    # Aggregation Task
    @task
    def aggregate_data():
        postgres_hook = PostgresHook("postgres_dibimbing").get_sqlalchemy_engine()

        # Simple aggregation logic, e.g., count users by signup date
        with postgres_hook.connect() as conn:
            df = pd.read_sql("SELECT extracted_at, COUNT(nama) as user_count FROM users GROUP BY extracted_at order by 1 desc", con=conn)
            df.to_sql("aggregates_result", con=conn, if_exists="replace", index=False)
        print("Data successfully aggregated and stored")

    aggregate_data_task = aggregate_data()

    start_task >> aggregate_data_task >> end_task

aggregation_dag = randa_dag_aggregate()
