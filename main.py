from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import joblib
import os
import shutil
import pandas as pd
from model_training import train_model
from pydantic import BaseModel
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename

    if not filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and XLSX files are allowed"
        )

    folder_path = f"datasets/1"

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "status": "Uploaded Succesfully"
    }

@app.get("/load/{dataset_id}")
def load(dataset_id: int):
    folder_path = f"datasets/1"

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
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")
    else:
        df = pd.read_excel(file_path)

    preview = df.head(5)
    head = preview.astype(object).where(pd.notna(preview), other=None).to_dict(orient="records")

    return {
        "dataset_id": 1,
        "file": data_file,
        "columns": list(df.columns),
        "head": head
    }

class TrainRequest(BaseModel):
    task: Literal["Classification", "Regression", "Clustering"]
    target: str | None = None

@app.post("/train/{dataset_id}")
def train(dataset_id: int, request: TrainRequest):
    task = request.task
    target = request.target

    folder_path = f"datasets/1"

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
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")
    else:
        df = pd.read_excel(file_path)

    if task == "Clustering":
        X = df
        y = None
    else:
        if target not in df.columns:
            raise HTTPException(status_code=400, detail="Target column not found")
        X = df.drop(columns=[target])
        y = df[target]
    
    results = train_model(X, y, task)

    joblib.dump({"model": results["model"], "label_encoder": results["le"]}, f"{folder_path}/final_model.joblib")

    return {
        "best_model": results.get("best_model_name"),
        "metrics": results.get("metrics"),
    }

@app.get("/download/{dataset_id}")
def download(dataset_id: int):
    file_path = f"datasets/1/final_model.joblib"

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

