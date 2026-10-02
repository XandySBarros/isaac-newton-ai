import os
from fastapi import FastAPI
from google import genai
from google.genai import types

app = FastAPI()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Personalize aqui a instrução do seu assistente de trabalho
SYSTEM_INSTRUCTION = """
Tu és o Isaac_Newton, um assistente virtual de elite focado em apoio técnico e profissional.
As tuas tarefas principais são:
1. Ajudar a estruturar relatórios técnicos e documentos de trabalho.
2. Resumir textos longos e extrair pontos de ação.
3. Responder de forma direta, clara, objetiva e profissional.
"""

@app.get("/")
def home():
    return {"status": "Isaac_Newton esta online e pronto para o trabalho!"}

@app.get("/chat")
def chat(prompt: str = "Ola"):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )
        return {"resposta": response.text}
    except Exception as e:
        return {"erro": str(e)}
