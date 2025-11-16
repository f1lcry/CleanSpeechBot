# syntax=docker/dockerfile:1
# TODO: Implement production-ready Dockerfile for the bot service.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "-m", "bot.main"]
