import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HF_TOKEN = os.getenv("HF_TOKEN")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Log in to Hugging Face
login(token=HF_TOKEN, add_to_git_credential=False)
logger.info("Logged in to Hugging Face")

# Configure the model for 4-bit quantization
model_id = "google/gemma-2b-it"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    logger.info("Model and tokenizer loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {str(e)}")
    raise e

# Define Pydantic models for input and output
class PredictInput(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

class PredictOutput(BaseModel):
    response: str
    prompt: str
    error: str | None = None

def generate_response(prompt: str, max_length: int, temperature: float, top_k: int, top_p: float) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return f"Error: {str(e)}"

app = FastAPI(title="Gemma AI API")

@app.post("/predict", response_model=PredictOutput)
async def predict(input_data: PredictInput):
    logger.info(f"Received request with payload: {input_data.dict()}")
    try:
        response = generate_response(
            prompt=input_data.prompt,
            max_length=input_data.max_length,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            top_p=input_data.top_p
        )
        return PredictOutput(
            response=response,
            prompt=input_data.prompt,
            error=None if not response.startswith("Error:") else response
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return PredictOutput(
            response="",
            prompt=input_data.prompt,
            error=str(e)
        )
        
def start_server():
    try:
        logger.info("Starting server without ngrok tunneling")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise e

if __name__ == "__main__":
    start_server()
