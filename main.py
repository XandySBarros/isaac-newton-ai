import os
import httpx
from fastapi import FastAPI, Request

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Armazenamento em memória do histórico por utilizador
conversation_history = {}
MAX_HISTORY_TURNS = 10 

SYSTEM_INSTRUCTION = """
Tu és o Isaac, um assistente virtual de elite focado em apoio técnico, robótica, automação e organização profissional.
Responda sempre de forma clara, objetiva, prestativa e profissional.
"""

# =====================================================================
# GESTÃO DE MEMÓRIA E HISTÓRICO
# =====================================================================
def get_messages_payload(chat_id: int, new_prompt: str):
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    
    # Adiciona histórico anterior
    history = conversation_history.get(chat_id, [])
    for msg in history:
        messages.append(msg)
        
    # Adiciona a mensagem atual
    messages.append({"role": "user", "content": new_prompt})
    return messages

def save_to_history(chat_id: int, user_text: str, bot_text: str):
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    
    conversation_history[chat_id].append({"role": "user", "content": user_text})
    conversation_history[chat_id].append({"role": "assistant", "content": bot_text})
    
    # Mantém o limite de histórico para não sobrecarregar
    if len(conversation_history[chat_id]) > MAX_HISTORY_TURNS * 2:
        conversation_history[chat_id] = conversation_history[chat_id][-(MAX_HISTORY_TURNS * 2):]

def clear_history(chat_id: int):
    if chat_id in conversation_history:
        conversation_history[chat_id] = []

# =====================================================================
# NÚCLEO CENTRAL DE IA (Groq - Llama 3.3 70B)
# =====================================================================
async def process_with_isaac(chat_id: int, prompt: str) -> str:
    messages = get_messages_payload(chat_id, prompt)
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(url, json=payload, headers=headers, timeout=30.0)
            if response.status_code == 200:
