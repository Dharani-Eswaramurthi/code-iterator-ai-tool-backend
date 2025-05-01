from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import re
import torch
from difflib import ndiff
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    BitsAndBytesConfig
)
from fastapi.middleware.cors import CORSMiddleware
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2560,
    temperature=0.2,
    top_p=0.9,
    do_sample=True,
    return_full_text=False,
    # device=0 if torch.cuda.is_available() else -1
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SuggestRequest(BaseModel):
    code: str
    prompt: str
    language: str

class SuggestResponse(BaseModel):
    improved_code: str
    explanation: str

def build_instruction(request: SuggestRequest) -> str:
    """Structured prompt with explicit formatting rules"""
    return f"""GAME_CODE_MODIFICATION

    You are an assisstant for game developers. Your task is to improve the productivity of game developers by providing code suggestions.
INSTRUCTIONS:
1. Analyze the provided code to verify it's actually {request.language}
2. If not in {request.language}, first convert it properly
3. Understand the request ( {request.prompt} ) and decide what to do with the code.
4. Do the changes as per the request, {request.prompt}
5. If the code is already done with the requested changes, do some improvements to the code and return.
6. Focus on game performance optimization
7. Use appropriate language conventions


Code to be modified: {request.code}

RESPONSE FORMAT (Provide only the JSON object and no other content, text, notes or any explanations):
{{ 
    "improved_code": "(this is the place for the improved code)",
    "explanation": "(This is the place for Short and crisp detailed technical justification)" 
}}

RULES:
- Return ONLY the JSON object with only the two mentioned keys and their respective values.
- Do not use any other characters or formatting
- Validate JSON syntax before responding
- Keep code concise and production-ready

NOTE: STRICTLY, Do not use any content or text outside the JSON object.

"""

def parse_model_output(raw: str) -> dict:
    """Extract the first valid JSON object from the output, even if surrounded by extra text."""
    try:
        # Find all potential JSON objects using a regex that handles some nested structures
        json_matches = re.findall(r'\{(?:[^{}]|(?:\{.*?\}))*\}', raw, re.DOTALL)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                # Check if both required keys exist and are strings
                if (isinstance(data.get("improved_code"), str) and 
                    isinstance(data.get("explanation"), str)):
                    return {
                        "improved_code": data["improved_code"].strip(),
                        "explanation": data["explanation"].strip()
                    }
            except json.JSONDecodeError:
                continue  # Skip invalid JSON
            except Exception:
                continue  # Other issues (e.g., type errors), try next match
        
        # If no valid partial JSON found, try parsing the entire output
        data = json.loads(raw)
        return {
            "improved_code": data.get("improved_code", raw).strip(),
            "explanation": data.get("explanation", "Optimization performed").strip()
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)} - Raw output: {raw[:200]}...")
        return {
            "improved_code": raw,
            "explanation": f"JSON Parsing Error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)} - Raw output: {raw[:200]}...")
        return {
            "improved_code": raw,
            "explanation": f"Processing Error: {str(e)}"
        }

@app.post("/suggest", response_model=SuggestResponse)
async def suggest(request: SuggestRequest) -> SuggestResponse:
    logger.info("Processing code suggestion request")
    start_time = time.time()
    
    try:
        instruction = build_instruction(request)
        
        with torch.inference_mode():
            outputs = generator(
                instruction,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=256
            )
            
        raw_output = outputs[0]['generated_text']
        logger.info(f"Raw model output: {raw_output}")  # Fixed logging syntax
        
        parsed = parse_model_output(raw_output)
        
        logger.info(f"Processing completed in {time.time() - start_time:.2f}s")
        return SuggestResponse(**parsed)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating suggestions: {str(e)}"
        )

def generate_safe_merge(original: str, modified: str) -> str:
    return '\n'.join(
        line[2:] for line in ndiff(
            original.split('\n'),
            modified.split('\n')
        ) if line.startswith(('+ ', '  '))
    )

class IntegrateRequest(BaseModel):
    original: str
    improved: str

@app.post("/integrate")
async def integrate(request: IntegrateRequest):
    try:
        merged_code = generate_safe_merge(request.original, request.improved)
        return {"integrated_code": merged_code}
    except Exception as e:
        logger.error(f"Integration error: {e}")
        raise HTTPException(status_code=400, detail="Failed to integrate changes")

@app.on_event("startup")
async def warmup_model():
    logger.info("Model warmup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)