# Полная последовательность установки Airflow нативно в WSL2 (для воспроизведения).
# Выполнять в WSL Ubuntu как root:  wsl -d Ubuntu -u root
# Файлы (airflow.env, *.sh) взять из этого каталога (airflow_wsl/), скопировать в /opt/airflow/.

set -e

# 1. Базовые пакеты + uv (менеджер Python)
apt-get update -qq
apt-get install -y -qq curl ca-certificates postgresql
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Python 3.12 (Ubuntu 26.04 несёт 3.14, несовместим с Airflow) + venv
uv python install 3.12
uv venv --python 3.12 /opt/airflow-venv

# 3. Airflow 2.10.5 + провайдеры (constraints для 3.12)
CONSTR="https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"
uv pip install --python /opt/airflow-venv/bin/python \
    "apache-airflow==2.10.5" \
    "apache-airflow-providers-postgres" \
    "apache-airflow-providers-ssh" \
    --constraint "$CONSTR"

# 4. PostgreSQL: роль + БД для метаданных Airflow
systemctl enable --now postgresql
su - postgres -c "psql -c \"CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow';\""
su - postgres -c "psql -c \"CREATE DATABASE airflow OWNER airflow;\""

# 5. Конфиг + инициализация
mkdir -p /opt/airflow
# (скопировать airflow.env из airflow_wsl/ в /opt/airflow/airflow.env)
source /opt/airflow/airflow.env
airflow db migrate
airflow users create --username admin --password admin \
    --firstname Admin --lastname User --role Admin --email admin@example.com

# 6. Импорт pools + variables (пути через glob, кириллица не печатается)
airflow pools import /mnt/d/Yandex.Disk/PYTHON/NLTK/*/airflow_setup/pools.json
airflow variables import /mnt/d/Yandex.Disk/PYTHON/NLTK/*/airflow_setup/variables_default.json

# 7. Запуск (nohup)
# (скопировать start_airflow.sh, stop_airflow.sh в /opt/airflow/, chmod +x)
/opt/airflow/start_airflow.sh

# Проверка:  curl http://localhost:8080/health  -> 200
