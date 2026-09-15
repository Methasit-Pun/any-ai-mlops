FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops ./mlops

CMD ["python", "-m", "mlops.log_experiment"]
