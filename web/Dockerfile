FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p static uploads dataset augmented_dataset model

EXPOSE 5000

CMD ["python", "main.py"]
