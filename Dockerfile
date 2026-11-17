FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data and the embedding index are built at runtime (or mounted) so the image
# stays small and the API key is never baked in:
#   docker run --env-file .env -v "$PWD/data:/app/data" -p 8501:8501 wnba-rag
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
