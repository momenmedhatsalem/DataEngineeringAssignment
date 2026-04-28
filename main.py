from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import joblib
import os
import pandas as pd
from contextlib import asynccontextmanager
from db import init_db, create_dataset, get_all_datasets
from model_training import train_model
from pydantic import BaseModel
from typing import Literal

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    init_db()

    yield  # app runs here

    # Shutdown logic (optional)

app = FastAPI(lifespan=lifespan)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename

    if not filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and XLSX files are allowed"
        )

    dataset_id = create_dataset()

    folder_path = f"datasets/{dataset_id}"
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "status": "Uploaded Succesfully"
    }

@app.get("/get_all")
def get_datasets():
    return {
        "datasets": get_all_datasets()
    }

@app.post("/load/{dataset_id}")
def load(dataset_id: int):
    folder_path = f"datasets/{dataset_id}"

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    files = os.listdir(folder_path)

    data_file = None
    for f in files:
        if f.endswith(".csv") or f.endswith(".xlsx"):
            data_file = f
            break

    if not data_file:
        raise HTTPException(status_code=404, detail="No CSV/XLSX file found")

    file_path = os.path.join(folder_path, data_file)

    if data_file.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    return {
        "dataset_id": dataset_id,
        "file": data_file,
        "columns": list(df.columns),
        "head": df.head(5).to_dict(orient="records")
    }

class TrainRequest(BaseModel):
    task: Literal["Classification", "Regression", "Clustering"]
    target: str | None = None

@app.post("/train/{dataset_id}")
def train(dataset_id: int, request: TrainRequest):
    task = request.task
    target = request.target

    folder_path = f"datasets/{dataset_id}"

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    files = os.listdir(folder_path)

    data_file = None
    for f in files:
        if f.endswith(".csv") or f.endswith(".xlsx"):
            data_file = f
            break

    if not data_file:
        raise HTTPException(status_code=404, detail="No CSV/XLSX file found")

    file_path = os.path.join(folder_path, data_file)

    if data_file.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Data Preprocessing Pipeline
    if task != "Clustering" and target not in df.columns:
        raise ValueError("Target column not found")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    results = train_model(X, y, task)

    joblib.dump(results["model"], f"{folder_path}/final_model.joblib")

    return {
        "best_model": results["best_model_name"]
    }

@app.get("/download/{dataset_id}")
def download(dataset_id: int):
    file_path = f"datasets/{dataset_id}/final_model.joblib"

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Model file not found"
        )

    return FileResponse(
        path=file_path,
        filename="final_model.joblib",
        media_type="application/octet-stream"
    )

