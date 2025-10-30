FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy code
COPY src/ /app/

# Create runtime dirs
RUN mkdir -p /app/data /app/artifacts /app/monitor

EXPOSE 8080 8001

CMD ["python", "service.py"]
