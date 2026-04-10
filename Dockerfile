FROM python:3.10-slim

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer cache optimisation)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=user . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 7860

# Start server, wait for it to be ready, then run inference
# `server` is the console_script defined in pyproject.toml
CMD server & \
    sleep 8 && \
    python -u inference.py; \
    wait
