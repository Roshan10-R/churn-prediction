FROM python:3.11-slim

WORKDIR /code

# Install dependencies first (better layer caching -- only re-runs
# pip install when requirements.txt actually changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the running API needs: the app, the src package
# (schemas import from src.config), and the trained model artifact.
COPY ./app ./app
COPY ./src ./src
COPY ./models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
