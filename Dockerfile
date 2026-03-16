FROM python:3.9-slim

WORKDIR /app

# Install system deps for Playwright/Chromium + pip deps in one layer
COPY requirements.txt ./
RUN apt-get update && apt-get install -y \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libxshmfence1 libx11-xcb1 \
    fonts-liberation libgl1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

COPY . .

EXPOSE 8001
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8001}
