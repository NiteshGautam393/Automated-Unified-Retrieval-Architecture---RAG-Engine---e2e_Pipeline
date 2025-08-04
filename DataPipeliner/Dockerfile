FROM apache/airflow:2.9.1

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt