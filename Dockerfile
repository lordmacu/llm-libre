FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY providers.yaml ./
RUN mkdir -p /datos
EXPOSE 8101
CMD ["uvicorn", "llm_libre.main:app", "--host", "0.0.0.0", "--port", "8101", "--workers", "1"]
