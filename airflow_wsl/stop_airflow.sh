#!/bin/bash
# Остановка Airflow (scheduler + webserver).
# Установить в /opt/airflow/stop_airflow.sh
pkill -f 'airflow webserver' 2>/dev/null || true
pkill -f 'airflow scheduler' 2>/dev/null || true
pkill -f gunicorn 2>/dev/null || true
sleep 2
rm -f /opt/airflow/airflow-webserver*.pid /opt/airflow/run/*.pid 2>/dev/null || true
echo "airflow stopped"
