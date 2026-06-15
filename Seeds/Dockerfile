FROM python:3.11-slim
WORKDIR /app_seeds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python",  "main.py"]