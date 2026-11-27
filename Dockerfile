FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentfund ./agentfund
RUN pip install --no-cache-dir .
ENTRYPOINT ["python", "-m", "agentfund"]
CMD ["compare"]
