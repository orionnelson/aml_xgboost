import os, json, asyncio, aiohttp
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
IN_TOPIC  = os.getenv("KAFKA_IN_TOPIC", "transactions")
OUT_TOPIC = os.getenv("KAFKA_OUT_TOPIC", "decisions")
SCORER_URL = os.getenv("SCORER_URL", "http://fraud-service:8080/score")

async def wait_for_tcp(host: str, port: int, timeout_s: int = 120):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    delay = 0.5
    while True:
        try:
            r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=3)
            w.close(); await w.wait_closed()
            return
        except Exception:
            if loop.time() > deadline: raise
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5.0)

async def start_clients():
    host, port = BOOTSTRAP.split(":")[0], int(BOOTSTRAP.split(":")[1])
    await wait_for_tcp(host, port)

    consumer = AIOKafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=True,
        group_id="fraud-scorer",
        metadata_max_age_ms=10000,
        request_timeout_ms=10000,
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
        request_timeout_ms=10000,
    )
    delay = 1.0
    while True:
        try:
            await consumer.start()
            await producer.start()
            return consumer, producer
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

async def run():
    consumer, producer = await start_clients()
    async with aiohttp.ClientSession() as session:
        try:
            async for msg in consumer:
                txn = msg.value
                async with session.post(SCORER_URL, json=txn, timeout=0.2) as resp:
                    decision = await resp.json()
                await producer.send_and_wait(OUT_TOPIC, decision)
        finally:
            await consumer.stop()
            await producer.stop()

if __name__ == "__main__":
    asyncio.run(run())
