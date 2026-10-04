FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    --break-system-packages \
    --ignore-installed cryptography \
    runpod

RUN python -m pip install --no-cache-dir \
    --break-system-packages \
    -r requirements.txt

COPY handler.py .
COPY prompts ./prompts

CMD ["python", "-u", "handler.py"]
