FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py /app/
COPY prompts /app/prompts

CMD ["python", "-u", "handler.py"]
