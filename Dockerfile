FROM runpod/pytorch:1.0.2-cuda12.8.1-torch2.8.0-ubuntu24.04

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY prompts ./prompts

CMD ["python", "-u", "handler.py"]