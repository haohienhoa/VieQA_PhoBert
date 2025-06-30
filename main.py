import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from contextlib import asynccontextmanager
from huggingface_hub import snapshot_download

CACHE_DIR = "/var/data/hf-cache"
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "phobert-vieqa-model")

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "haohahahihihehe/PhoBert_VieQA")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_CACHE_PATH
    
    if not os.path.exists(MODEL_CACHE_PATH):
        print(f"Model not found in cache. Downloading model '{HF_MODEL_ID}' to '{MODEL_CACHE_PATH}'...")
        os.makedirs(CACHE_DIR, exist_ok=True) 
        snapshot_download(repo_id=HF_MODEL_ID, local_dir=MODEL_CACHE_PATH)
        print("Model downloaded successfully.")
    else:
        print(f"Model found in cache at '{MODEL_CACHE_PATH}'. Loading from cache.")
        
    print("Loading model and tokenizer from local path...")
    models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_CACHE_PATH)
    models['model'] = AutoModelForQuestionAnswering.from_pretrained(MODEL_CACHE_PATH).to(DEVICE)
    models['model'].eval()
    print("Model loaded successfully into memory!")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QuestionRequest(BaseModel):
    context: str
    question: str

class AnswerResponse(BaseModel):
    answer: str

def get_prediction(question: str, context: str) -> str:
    tokenizer = models['tokenizer']
    model = models['model']
    
    inputs = tokenizer(question, context, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)
    
    if end_index < start_index:
        return ""

    answer_tokens = inputs["input_ids"][0, start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer.strip()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=AnswerResponse)
def predict(request: QuestionRequest):
    answer = get_prediction(request.question, request.context)
    return AnswerResponse(answer=answer)