FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# default: HTTP interface on port 8000 (POST an observation to /act)
EXPOSE 8000
ENTRYPOINT ["python", "main.py"]
CMD ["serve", "--spec", "examples/spec.json", "--port", "8000"]
