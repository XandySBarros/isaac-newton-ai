import os
from fastapi import FastAPI
from google import genai

app = FastAPI()

# Inicializa o cliente do Gemini usando a chave cadastrada nas variáveis do servidor
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.get("/")
def home():
    return {"status": "Isaac_Newton esta online!"}

@app.get("/chat")
def chat(prompt: str = "Ola"):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        return {"resposta": response.text}
    except Exception as e:
        return {"erro": str(e)}
