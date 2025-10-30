
# Fraud-RT: Real-Time Fraud Detection with Kafka, Redis, and XGBoost

This project is a **real-time fraud detection pipeline** using:
- **Apache Kafka** (event streaming)
- **Zookeeper** (Kafka coordination)
- **Redis** (caching / fast lookup)
- **XGBoost** (fraud detection model)
- **Prometheus** (metrics)
- **Grafana** (dashboards)
- **FastAPI** (fraud scoring service)

---

## Features
- Ingests synthetic or real transactions via **Kafka**.
- Fraud scoring service (`fraud-service`) serves requests on `http://localhost:8080/score`.
- **Trainer** auto-generates synthetic data and trains an XGBoost model.
- **Consumer** listens to Kafka topics and forwards fraud decisions.
- **Prometheus + Grafana** monitoring for service metrics and dashboards.
- Modular, containerized architecture using `docker-compose`.

---

## Project Structure
```

fraud-rt/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── src/
│   ├── service.py          # FastAPI fraud scoring service
│   ├── consumer.py         # Kafka consumer
│   ├── train.py            # XGBoost trainer
│   └── utils.py            # Helpers
├── data/
│   └── make_synth.py       # Synthetic dataset generator
├── grafana/
│   ├── dashboards/
│   └── provisioning/
│       ├── dashboards/
│       └── datasources/
├── prometheus.yml
├── artifacts/              # Saved models
├── monitor/                # Metrics logs
└── postman/
└── Postman Metrics.png # Example metrics screenshot

````

---

## Running Locally

### 1. Start all services
```bash
docker compose up --build
````

This will spin up:

* `redis:latest`
* `zookeeper:3.9`
* `apache/kafka:3.8.0`
* `fraud-service`
* `fraud-consumer`
* `fraud-trainer`
* `prometheus`
* `grafana`

### 2. Check service health

```bash
docker compose ps
docker logs fraud-service
docker logs fraud-consumer
docker logs fraud-trainer
```

### 3. Test fraud scoring API

Linux/Mac:

```bash
curl -s http://localhost:8080/score -H 'Content-Type: application/json' \
-d '{"request_id":"txn_123","account_id":"acct_42","merchant_category":"electronics","amount":129.99,"device_trust_score":0.7,"ip_risk_score":0.2,"acct_age_days":380,"txns_last_5m":2,"declines_last_24h":0,"chargebacks_90d":0}'
```

Windows CMD:

```cmd
curl -s http://localhost:8080/score -H "Content-Type: application/json" -d "{\"request_id\":\"txn_123\",\"account_id\":\"acct_42\",\"merchant_category\":\"electronics\",\"amount\":129.99,\"device_trust_score\":0.7,\"ip_risk_score\":0.2,\"acct_age_days\":380,\"txns_last_5m\":2,\"declines_last_24h\":0,\"chargebacks_90d\":0}"
```

### 4. Access dashboards

* Fraud Service API: [http://localhost:8080/docs](http://localhost:8080/docs)
* Prometheus: [http://localhost:9090](http://localhost:9090)
* Grafana: [http://localhost:3000](http://localhost:3000) (default: `admin/admin`)

---

## Metrics Screenshot

Example Postman metrics visualization:

![Postman Metrics](postman/Postman%20Metrics.png)

---

## Environment Variables

Fraud service:

* `MODEL_PATH=/app/artifacts/model_v1.xgb`
* `KAFKA_BOOTSTRAP=kafka:9092`
* `KAFKA_IN_TOPIC=transactions`
* `KAFKA_OUT_TOPIC=decisions`
* `REDIS_URL=redis://redis:6379/0`
* `PROMETHEUS_PORT=8001`

Kafka:

* `KAFKA_CFG_ZOOKEEPER_CONNECT=zookeeper:2181`
* `KAFKA_CFG_LISTENERS=PLAINTEXT://:9092`
* `KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092`

---

## Example Flow

1. `trainer` generates synthetic data and trains XGBoost.
2. Model saved to `artifacts/model_v1.xgb`.
3. `fraud-service` loads the model and exposes `/score`.
4. Transactions sent to Kafka `transactions` topic.
5. `stream-consumer` consumes messages, requests fraud decision, and publishes results to `decisions`.
6. Metrics exposed to **Prometheus** and visualized in **Grafana**.

---

## Troubleshooting

* If Kafka fails with `Missing required configuration zookeeper.connect`, ensure `KAFKA_ENABLE_KRAFT=false`.
* If consumer errors with `GroupCoordinatorNotAvailableError`, wait a few seconds — the `__consumer_offsets` topic must initialize.
* To rebuild from scratch:

```bash
docker compose down -v --remove-orphans
docker compose up --build
```

---

## License

MIT

```
```
