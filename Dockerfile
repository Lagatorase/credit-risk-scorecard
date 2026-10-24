FROM python:3.12-slim

WORKDIR /app

# LightGBM needs the OpenMP runtime, which the slim image doesn't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scorecard ./scorecard
COPY config ./config
COPY api ./api
COPY app ./app
COPY docs ./docs
COPY .streamlit ./.streamlit

# the trained model, its reference data and the demo applicants are built on my machine and mounted in,
# because they come from Home Credit's data and aren't in the repo
VOLUME ["/app/models"]
EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
