#!/bin/bash
# Запуск Airflow (scheduler + webserver) в фоне через nohup.
# Airflow в WSL2, тяжёлая работа — на Windows через WSL-interop (BashOperator).
# Установить в /opt/airflow/start_airflow.sh
set -e
source /opt/airflow/airflow.env
mkdir -p /opt/airflow/run /opt/airflow/logs

# Остановить прошлые процессы, если висят
pkill -9 -f 'airflow webserver' 2>/dev/null || true
pkill -9 -f 'airflow scheduler' 2>/dev/null || true
pkill -9 -f gunicorn 2>/dev/null || true
sleep 2
rm -f /opt/airflow/airflow-webserver*.pid

# Scheduler
nohup airflow scheduler > /opt/airflow/logs/scheduler.out 2>&1 &
echo $! > /opt/airflow/run/scheduler.pid

# Webserver
nohup airflow webserver --port 8080 > /opt/airflow/logs/webserver.out 2>&1 &
echo $! > /opt/airflow/run/webserver.pid

echo "scheduler pid=$(cat /opt/airflow/run/scheduler.pid) webserver pid=$(cat /opt/airflow/run/webserver.pid)"
echo "logs: /opt/airflow/logs/{scheduler,webserver}.out"
