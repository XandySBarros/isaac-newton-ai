import os
import httpx
from fastapi import FastAPI, Request

app = FastAPI()

# Variáveis de ambiente configuradas no Render
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# Histórico de conversas em memória por chat_id
conversation_history = {}
MAX_HISTORY_TURNS = 10 

SYSTEM_INSTRUCTION = """
Você é o JARVIS, um assistente virtual de elite focado em apoio técnico, robótica, automação, programação (ESP32, Arduino) e organização de projetos.
Responda sempre de forma clara, objetiva, prestativa e profissional.
"""

def get_gemini_contents(chat_id: int, new_prompt: str):
    contents = []
    history = conversation_history.get(chat_id, [])
    
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["text"]}]})
        
    contents.append({"role": "user", "parts": [{"text": new_prompt}]})
    return contents

def save_to_history(chat_id: int, user_text: str, bot_text: str):
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    conversation_history[chat_id].append({"role": "user", "text": user_text})
    conversation_history[chat_id].append({"role": "model", "text": bot_text})
    
    # Mantém apenas as últimas interações na memória
    if len(conversation_history[chat_id]) > MAX_HISTORY_TURNS * 2:
        conversation_history[chat_id] = conversation_history[chat_id][-(MAX_HISTORY_TURNS * 2):]

def clear_history(chat_id: int):
    if chat_id in conversation_history:
        conversation_history[chat_id] = []

async def process_with_jarvis(chat_id: int, prompt: str) -> str:

    print("========================================")
    print("JARVIS: iniciando chamada ao Gemini")
    print(f"API KEY configurada: {bool(GEMINI_API_KEY)}")
    print(f"Tamanho da API KEY: {len(GEMINI_API_KEY)}")
    print(f"Mensagem recebida: {prompt}")
    print("========================================")

    if not GEMINI_API_KEY:
        print("ERRO: GEMINI_API_KEY NÃO CONFIGURADA")
        return "⚠️ A GEMINI_API_KEY não está configurada no Render."

    contents = get_gemini_contents(chat_id, prompt)

    model_name = "gemini-3.8-flash"

    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model_name}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [
                {
                    "text": SYSTEM_INSTRUCTION
                }
            ]
        }
    }

    try:

        print(f"JARVIS: chamando modelo {model_name}")
        print(f"JARVIS: URL = {url}")

        async with httpx.AsyncClient() as http_client:

            response = await http_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=60.0
            )

        print(f"JARVIS: Gemini respondeu HTTP {response.status_code}")
        print(f"JARVIS: resposta = {response.text[:2000]}")

        if response.status_code == 200:

            data = response.json()

            bot_reply = (
                data["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

            print("JARVIS: resposta recebida com sucesso!")

            save_to_history(
                chat_id,
                prompt,
                bot_reply
            )

            return bot_reply

        return (
            f"⚠️ Gemini retornou HTTP "
            f"{response.status_code}.\n\n"
            f"Veja os detalhes no log do Render."
        )

    except Exception as e:

        print("========================================")
        print("ERRO AO CHAMAR GEMINI")
        print(repr(e))
        print("========================================")

        return (
            "⚠️ Erro ao se comunicar com o Gemini. "
            "Veja o log do Render."
        )
@app.get("/")
def home():
    return {"status": "JARVIS está online e operacional!"}

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"status": "invalid json"}

    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"].strip()
        
        if user_message == "/start":
            clear_history(chat_id)
            reply = (
                "🤖 **Olá! Eu sou o JARVIS.**\n\n"
                "Estou pronto para ajudar com robótica, automação, programação e desenvolvimento de projetos.\n\n"
                "💡 **Dica:** Para reiniciar nossa conversa e limpar a memória, envie `/limpar`."
            )
        elif user_message in ["/limpar", "/reset"]:
            clear_history(chat_id)
            reply = "🧹 **Memória limpa com sucesso!** Podemos começar um novo assunto."
        else:
            reply = await process_with_jarvis(chat_id, user_message)
        
        if TELEGRAM_BOT_TOKEN:
            telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as http_client:
                await http_client.post(telegram_url, json={
                    "chat_id": chat_id,
                    "text": reply,
                    "parse_mode": "Markdown"
                })
                
    return {"status": "ok"}
