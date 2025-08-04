from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


project_directory = "/opt/airflow"

default_args = {
    'owner': 'Nitesh Gautam',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 1),
    'email': ['niteshgautam19118072@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1, seconds=30),
    'execution_timeout': timedelta(minutes=10),
}

dag = DAG(
    'bbc_scraper_etl_pipeline',
    default_args=default_args,
    description='Scrape BBC news for ETL pipeline and RAG',
    schedule_interval=None, 
    catchup=False,
    tags=['scraper', 'bbc', 'etl', 'rag']
)


RunScraper = BashOperator(
    task_id='Scraper',
    bash_command=f'python {project_directory}/dags/Scraper.py',
    dag=dag
)

BronzeTransformation = BashOperator(
    task_id='Bronze_Transform',
    bash_command=f'python {project_directory}/Transforms/To_Bronze.py',
    dag=dag
)

SilverTransformation = BashOperator(
    task_id='Silver_Transform',
    bash_command=f'python {project_directory}/Transforms/To_Silver.py',
    dag=dag
)

GoldTransformation = BashOperator(
    task_id='Gold_Transform',
    bash_command=f'python {project_directory}/Transforms/To_Gold.py',
    dag=dag
)

ChromaTransformation = BashOperator(
    task_id='Chroma_Transform',
    bash_command=f'python {project_directory}/Transforms/To_Chroma.py',
    dag=dag
)

# Task dependencies
RunScraper >> BronzeTransformation >> SilverTransformation >> GoldTransformation >> ChromaTransformation