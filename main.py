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

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "haohahahihihehe/PhoBert_VieQA")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"No persistent disk detected. Downloading model '{HF_MODEL_ID}' on every startup...")
    
    model_path = snapshot_download(repo_id=HF_MODEL_ID)
    
    print(f"Model downloaded to temporary path: {model_path}")
    print("Loading model and tokenizer from temporary path...")
    
    models['tokenizer'] = AutoTokenizer.from_pretrained(model_path)
    models['model'] = AutoModelForQuestionAnswering.from_pretrained(model_path).to(DEVICE)
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
