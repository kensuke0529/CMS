FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
COPY app/ app/

CMD ["bash", "-c", "python pipeline_wo_aws.py && streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0"]
