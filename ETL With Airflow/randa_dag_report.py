from airflow.decorators import dag, task
from airflow.utils.context import Context
from airflow.utils.email import send_email
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowFailException
from datetime import datetime
import pandas as pd
from airflow.operators.empty import EmptyOperator
from airflow.utils.context import Context
from airflow.providers.postgres.hooks.postgres import PostgresHook

@dag(
    schedule_interval="0 9 * * *",  # Runs daily at 9 AM
    start_date=datetime(2023, 9, 5),
    catchup=False,
    tags=["send_report"],
)
def randa_dag_report():
    # Start and End tasks
    start_task = EmptyOperator(task_id="start_task")
    end_task = EmptyOperator(
        task_id="end_task",
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Run only if previous tasks are successful
    )

    # Task to generate the report and send email
    @task
    def generate_and_send_report():
        # Connect to Postgres to get the latest analysis results
        postgres_hook = PostgresHook("postgres_dibimbing")
        conn = postgres_hook.get_conn()
        cursor = conn.cursor()
        
        # Query to get the latest analysis results
        query = """
        SELECT * FROM aggregates_result;
        """
        cursor.execute(query)
        result = cursor.fetchall()

        # Format the result as needed
        # Example: Convert to a DataFrame or a string
        df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
        report_content = df.to_html()

        # Send the email
        send_email(
            to=["randayandika1@gmail.com"],
            subject="Report Daily",
            html_content=f"""
                <center><h1>Report Daily</h1></center>
                <p>The latest analysis results:</p>
                {report_content}
            """
        )
        print("Analysis report successfully sent via email.")

    # Task dependencies
    start_task >> generate_and_send_report() >> end_task

# Instantiate the DAG
send_report_dag_instance = randa_dag_report()
