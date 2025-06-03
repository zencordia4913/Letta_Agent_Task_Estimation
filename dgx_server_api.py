from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import logging

# FastAPI
app = FastAPI()

# Logging 
logging.basicConfig(level=logging.INFO)

# GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# MiniLM Model & Tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)


class TaskData(BaseModel):
    data: list  # Dictionaries with "title" and "task_duration_hours"

class TaskName(BaseModel):
    task_name: str

# Regression Model
class MiniLMRegressor(nn.Module):
    def __init__(self, base_model):
        super(MiniLMRegressor, self).__init__()
        self.base_model = base_model
        self.regressor = nn.Linear(384, 1)  

        # Unfreeze last 2 transformer layers
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze all first
        for param in list(self.base_model.encoder.layer[-2:].parameters()):
            param.requires_grad = True  # Unfreeze last two layers

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        return self.regressor(embeddings).squeeze(1)

# Model & Optimizer
model = MiniLMRegressor(base_model).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
criterion = nn.HuberLoss()

# Training
@app.post("/train")
async def train_model(task: TaskData):
    """Receives task data, processes it, and trains a MiniLM model."""
    try:
        # Convert JSON to df
        df = pd.DataFrame(task.data)

        if "title" not in df.columns or "task_duration_hours" not in df.columns:
            raise HTTPException(status_code=400, detail="Missing required fields: 'title' or 'task_duration_hours'")

        logging.info(f"Received {len(df)} records for training.")

        # Normalize Task Duration
        mean_task_duration = df["task_duration_hours"].mean()
        std_task_duration = df["task_duration_hours"].std()
        df["task_duration_hours"] = (df["task_duration_hours"] - mean_task_duration) / std_task_duration

        # df to hugging face df
        dataset = Dataset.from_pandas(df)

        # tokenization
        def tokenize(batch):
            return tokenizer(batch["title"], padding="max_length", truncation=True, return_tensors="pt")

        encoded_dataset = dataset.map(tokenize, batched=True)

        # Input tensors
        input_ids = torch.stack([torch.tensor(x) for x in encoded_dataset["input_ids"]]).squeeze(1).to(device)
        attention_mask = torch.stack([torch.tensor(x) for x in encoded_dataset["attention_mask"]]).squeeze(1).to(device)
        labels = torch.tensor(df["task_duration_hours"].values, dtype=torch.float32).to(device)

        # PyTorch DataLoader
        train_dataset = TensorDataset(input_ids, attention_mask, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Training Loop
        num_epochs = 15
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logging.info(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_dataloader)}")

        # Save Model
        model_save_path = "/data/students/jeryl/OMNI/Omni"
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_save_path}/model.pt")
        tokenizer.save_pretrained(model_save_path)

        logging.info(f"Model saved to {model_save_path}")


        return {"message": "Model trained successfully", "model_path": model_save_path}

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        return HTTPException(status_code=500, detail=str(e))



@app.post("/inference")
async def run_inference(task: TaskData):
    """Returns embeddings for task titles with duration as JSON."""
    try:
        df = pd.DataFrame(task.data)

        if "title" not in df.columns or "task_duration_hours" not in df.columns:
            raise HTTPException(status_code=400, detail="Missing required fields: 'title' or 'task_duration_hours'")

        logging.info(f"Received {len(df)} records for inference.")

        # Load trained model weights
        model_path = "/data/students/jeryl/OMNI/Omni"
        model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
        model.eval()

        # Tokenize input
        titles = df["title"].tolist()
        batch_size = 128
        embeddings_list = []


        # Get embeddings (remove regression head)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(titles), batch_size):
                batch_titles = titles[i:i+batch_size]

                encoded = tokenizer(batch_titles, padding="max_length", truncation=True, return_tensors="pt").to(device)
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS

                # Move to CPU and convert to list
                embeddings_list.extend(batch_embeddings.cpu().numpy().tolist())

        # Package as JSON
        result = []
        for i in range(len(df)):
            result.append({
                "title": df.iloc[i]["title"],
                "task_duration_hours": df.iloc[i]["task_duration_hours"],
                "embedding": embeddings_list[i]
            })

        return result

    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prod_infer")
async def prod_inference(task: TaskName):
    """Returns embeddings for task titles."""
    try:
        if not isinstance(task.task_name, str):
            raise HTTPException(status_code=400, detail="Invalid type for 'task_name'. Must be a string.")

        logging.info("Received task name for inference.")

        # Load trained model weights
        model_path = "/data/students/jeryl/OMNI/Omni"
        model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
        model.eval()

        encoded = tokenizer(task.task_name, padding="max_length", truncation=True, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
        return embeddings.detach().cpu().numpy().tolist()

    except HTTPException as e:
        raise e  # Let FastAPI handle the intended exception

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Start FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
