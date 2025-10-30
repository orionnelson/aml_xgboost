from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "artifacts/model_v1.xgb"
    model_version: str = "v1"
    redis_url: str = "redis://redis:6379/0"
    kafka_bootstrap: str = "kafka:9092"
    kafka_in_topic: str = "transactions"
    kafka_out_topic: str = "decisions"
    cache_ttl_seconds: int = 300
    service_name: str = "fraud-service"
    prometheus_port: int = 8001

settings = Settings(_env_file=None)
