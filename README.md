# mlops-flow-1

echo AIRFLOW_UID=50000 > .env

printf "PROJECT_ROOT=%s\n" "$(realpath ..)" > .env

docker compose build
docker compose up airflow-init
docker compose up -d

mlflow ui --host 0.0.0.0 --port 5000


