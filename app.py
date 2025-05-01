from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi.middleware.cors import CORSMiddleware

# Model setup
model_name = "stabilityai/stable-code-instruct-3b"
assert torch.cuda.is_available(), "CUDA not detected—check your torch install!"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=4096,
    temperature=0.2
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

@app.post("/suggest", response_model=SuggestResponse)
def suggest(request: SuggestRequest) -> SuggestResponse:
    print("Received /suggest request")
    print("Request code:", request.code)
    print("Request prompt:", request.prompt)
    print("Request language:", request.language)
    # Clear GPU cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache")
        torch.cuda.empty_cache()

    # Build instruction
    instruction = (
        f"Here is the code to improve in {request.language}:```{request.code}```"
        f"Please apply this change: {request.prompt}. "
        "Return ONLY a single JSON object with exactly two keys:"
        "\n  - \"improved_code\": the improved code as a raw string,"
        "\n  - \"explanation\": the explanation as a raw string."
        " Do NOT include any other text or markdown."
    )
    print("Instruction sent to model:", instruction)

    outputs = generator(instruction)
    print("Model outputs:", outputs)
    raw = outputs[0].get("generated_text", "")
    print("RAW output from model:", raw)

    # Robust JSON extraction
    code, exp = None, None
    try:
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(raw):
            try:
                payload, end = decoder.raw_decode(raw[idx:])
                print("Decoded JSON candidate:", payload)
                if isinstance(payload, dict) and "improved_code" in payload and "explanation" in payload:
                    code = payload["improved_code"].strip()
                    exp = payload["explanation"].strip()
                    print("Parsed JSON successfully")
                    break
            except json.JSONDecodeError:
                idx += 1
                continue
    except Exception as e:
        print("Exception during JSON decode:", e)

    # Fallback: regex-based
    if code is None or exp is None:
        print("Trying regex-based JSON extraction")
        match = re.search(r"\{\s*\"improved_code\".*\}" , raw, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                print("Regex-extracted JSON:", payload)
                code = payload.get("improved_code", "").strip()
                exp = payload.get("explanation", "").strip()
            except json.JSONDecodeError as e:
                print("Regex JSON decode error:", e)

    # Final fallback: return raw
    if code is None or exp is None:
        print("Falling back to raw output")
        code = raw
        exp = ""

    print("Final improved_code:", code)
    print("Final explanation:", exp)
    return SuggestResponse(improved_code=code, explanation=exp)

class IntegrateRequest(BaseModel):
    original: str
    improved: str

@app.post("/integrate", response_model=str)
def integrate(request: IntegrateRequest):
    """
    Integrate improved code into original using unified diff merge.
    """
    print("Received /integrate request")
    # For now, simply return the improved code; you can enhance with three-way merge later
    return request.improved