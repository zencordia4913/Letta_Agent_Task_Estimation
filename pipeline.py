import sys
import os
import django
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json

# FastAPI Init
app = FastAPI()

# DGX Server API Endpoint
DGX_API_URL_TRAIN = os.getenv("DGX_API_URL_TRAIN") 
DGX_API_URL_INF = "http://202.92.159.242:8001/inference"

# Django sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "src/plex"))
sys.path.insert(0, os.path.join(BASE_DIR, "src/sileo"))
sys.path.insert(0, os.path.join(BASE_DIR, "deps/bots"))
sys.path.insert(0, os.path.join(BASE_DIR, "deps/unzipper"))

# Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "centralbpo.settings")
django.setup()

print("LOADING HISTORICAL TASK DATA...")

# Django model imports
from hqzen.models import HistoricalTask

# Extracting Data
def fetch_historical_data():
    """Query data from Django models and convert to Pandas DataFrame."""
    tasks = HistoricalTask.objects.all()[:100000]
    field_names = [field.name for field in HistoricalTask._meta.get_fields()]
    task_data = [{field: getattr(task, field, "") for field in field_names} for task in tasks]
    
    return pd.DataFrame(task_data)

df = fetch_historical_data()

# Data Cleanup and Transformations (Adding Duration)
def preprocess_data(df):
    """Cleans and transforms the DataFrame."""
    date_columns = ["date_created", "done_at"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Duration Computation
    df["duration"] = df["done_at"] - df["date_created"]
    df["duration"] = df["duration"].fillna(pd.Timedelta(seconds=0))
    df["task_duration_days"] = df["duration"].dt.days
    df["task_duration_hours"] = df["duration"].dt.total_seconds() / 3600
    df["task_duration_minutes"] = df["duration"].dt.total_seconds() / 60

    return df[df["done_at"].notna()]  # Filter out unfinished tasks

print("PREPROCESSING DATA")
df = preprocess_data(df)
print("DATA CLEANED & READY FOR TRAINING")
print(df.head())

# FastAPI Data Model
class TaskData(BaseModel):
    text: str  # Task Name
    labels: list  # Task Duration

@app.post("/send-data/")
def send_data():
    """Sends cleaned df rows to DGX Server for training."""
    try:
        data_json = df[["title", "task_duration_hours"]].to_json(orient="records") # DataFrame to JSON
        response = requests.post(DGX_API_URL_TRAIN, json={"data": json.loads(data_json)}) # Send data to DGX

        print(f"Response from DGX: {response.status_code} - {response.text}")

        if response.status_code == 200:
            return {"message": "Model successfully trained", "response": response.json()}
        else:
            raise HTTPException(status_code=response.status_code, detail="DGX Training Error")
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get-embeddings/")
def send_data():
    """Sends cleaned df rows to DGX Server for inferencing."""
    import psycopg2
    from psycopg2.extras import execute_values
    try:
        print("FEEDING DATA TO INFERENCE SERVER")
        data_json = df[["title", "task_duration_hours"]].to_json(orient="records")
        response = requests.post(
            DGX_API_URL_INF, 
            json={"data": json.loads(data_json)}
        )
        inference_data = response.json()
        print(f"Response from DGX: {response.status_code}")
        print("RECEIVED EMBEDDINGS FROM SERVER")
        conn = psycopg2.connect(
            dbname="omni_data",
            user="jeryl4913",
            password=os.getenv("DB_PASSWORD"),  
            host="localhost",
            port="5433"
        )
        cur = conn.cursor()
        print("STORING DATA")

        unique_data = {}
        for task in inference_data:
            unique_data[task["title"]] = task

        records = [
            (t["title"], t["task_duration_hours"], t["embedding"])
            for t in unique_data.values()
        ]

        # psycopg2 bulk insert
        query = """
            INSERT INTO task_embeddings (title, task_duration_hours, embedding)
            VALUES %s
            ON CONFLICT (title)
            DO UPDATE SET
                task_duration_hours = EXCLUDED.task_duration_hours,
                embedding = EXCLUDED.embedding;
        """

        execute_values(cur, query, records)
        conn.commit()
        cur.close()
        conn.close()

        print("SUCCESSFULLY STORED DATA TO PSQL.")


        if response.status_code == 200:
            return {"message": "Task data successfully converted to embeddings", "response": response.json()}
        else:
            raise HTTPException(status_code=response.status_code, detail="DGX Training Error")
            
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        


# FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


