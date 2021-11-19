from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn

app = FastAPI()
tokenizer_en2es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en2es = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokenizer_es2en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
es2en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")


@app.get("/to_english/{prompt}")
async def to_english(prompt: str) -> dict:
    input_ids = tokenizer_es2en.encode(prompt, return_tensors="pt")
    outputs = es2en.generate(input_ids)
    translated_result = tokenizer_es2en.decode(outputs[0], skip_special_tokens=True)
    return {"translated": translated_result}


@app.get("/to_spanish/{prompt}")
async def to_spanish(prompt: str) -> dict:
    input_ids = tokenizer_en2es.encode(prompt, return_tensors="pt")
    outputs = en2es.generate(input_ids)
    translated_result = tokenizer_en2es.decode(outputs[0], skip_special_tokens=True)
    return {"translated": translated_result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
