FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uv

COPY --chown=user . .

RUN pip install -e .

EXPOSE 7860

CMD server & sleep 5 && python -u inference.py && wait