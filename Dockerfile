FROM python:3.12-slim

WORKDIR /app
COPY . /app

EXPOSE 8765
VOLUME ["/data"]

CMD ["python", "-m", "openagent", "--host", "0.0.0.0", "--port", "8765", "--data-dir", "/data"]

