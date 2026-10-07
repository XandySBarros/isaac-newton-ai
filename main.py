import asyncio
import os

import httpx
from fastapi import BackgroundTasks, FastAPI, Request

app = FastAPI()

# ---------------------------------------------------------------------------
# Configuração (variáveis de ambiente do Render)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Ordem: modelos menos disputados primeiro, para evitar o erro 503.
MODELS_TO_TRY = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

# Personalidade / regras do bot. Edite como quiser.
SYSTEM_INSTRUCTION = (
    "Você é o JARVIS, um assistente pessoal inteligente que responde "
    "comandos e perguntas pelo Telegram. Responda sempre em português, "
    "de forma clara, objetiva e amigável. Ajude também com projetos de "
    "eletrônica e programação, como ESP32."
)

# ---------------------------------------------------------------------------
# Memória de conversa (em RAM: zera quando o Render reinicia)
# ---------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 20
chat_histories: dict[int, list[dict]] = {}

# Evita processar a mesma atualização do Telegram duas vezes
processed_updates: set[int] = set()


def get_gemini_contents(chat_id: int, prompt: str) -> list[dict]:
    """Histórico da conversa + a nova mensagem do usuário."""
    history = chat_histories.get(chat_id, [])
    return history + [{"role": "user", "parts": [{"text": prompt}]}]


def save_to_history(chat_id: int, prompt: str, reply: str) -> None:
    history = chat_histories.setdefault(chat_id, [])
    history.append({"role": "user", "parts": [{"text": prompt}]})
    history.append({"role": "model", "parts": [{"text": reply}]})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]


# ---------------------------------------------------------------------------
# Gemini (com retry e fallback entre modelos)
# ---------------------------------------------------------------------------
async def process_with_jarvis(chat_id: int, prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ A GEMINI_API_KEY não está configurada no Render."

    contents = get_gemini_contents(chat_id, prompt)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
    }

    last_status = None

    async with httpx.AsyncClient(timeout=45.0) as client:
        for model in MODELS_TO_TRY:
            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{model}:generateContent"
            )

            for _attempt in range(2):  # 2 tentativas por modelo
                try:
                    r = await client.post(url, json=payload, headers=headers)
                except Exception as e:
                    print(f"JARVIS: {model} erro de rede: {repr(e)}")
                    last_status = "timeout"
                    await asyncio.sleep(1.5)
                    continue

                print(f"JARVIS: {model} -> HTTP {r.status_code}")

                if r.status_code == 200:
                    try:
                        parts = r.json()["candidates"][0]["content"]["parts"]
                        reply = "".join(p.get("text", "") for p in parts).strip()
                    except Exception:
                        print(f"JARVIS: resposta inesperada: {r.text[:1000]}")
                        break  # tenta o próximo modelo

                    if not reply:
                        print(f"JARVIS: resposta vazia: {r.text[:1000]}")
                        break

                    save_to_history(chat_id, prompt, reply)
                    return reply

                last_status = r.status_code
                print(f"JARVIS: {r.text[:500]}")

                if r.status_code in (429, 500, 503):
                    await asyncio.sleep(1.5)  # congestionado: tenta de novo
                    continue

                break  # 400/401/403/404: vai para o próximo modelo

    return (
        f"⚠️ Todos os modelos falharam (último erro: {last_status}). "
        "Tente de novo em instantes."
    )


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
async def send_telegram_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("ERRO: TELEGRAM_BOT_TOKEN não configurado no Render.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # O Telegram aceita no máximo 4096 caracteres por mensagem
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [""]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            try:
                r = await client.post(url, json={"chat_id": chat_id, "text": chunk})
                if r.status_code != 200:
                    print(f"TELEGRAM: erro {r.status_code}: {r.text[:500]}")
            except Exception as e:
                print(f"TELEGRAM: falha ao enviar: {repr(e)}")


async def handle_update(update: dict) -> None:
    """Processa uma mensagem em segundo plano."""
    try:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        chat_id = message["chat"]["id"]
        user_message = message.get("text")

        if not user_message:
            await send_telegram_message(
                chat_id, "Por enquanto só consigo ler mensagens de texto. 🙂"
            )
            return

        if user_message.strip() == "/start":
            await send_telegram_message(
                chat_id, "Olá! Eu sou o JARVIS. Pode me mandar seus comandos. 🤖"
            )
            return

        if user_message.strip() == "/limpar":
            chat_histories.pop(chat_id, None)
            await send_telegram_message(chat_id, "Memória da conversa apagada. ✅")
            return

        print(f"JARVIS: mensagem recebida de {chat_id}: {user_message[:200]}")
        reply = await process_with_jarvis(chat_id, user_message)
        await send_telegram_message(chat_id, reply)

    except Exception as e:
        print(f"ERRO em handle_update: {repr(e)}")


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    update = await request.json()
    update_id = update.get("update_id")

    # Telegram reenvia a mesma atualização se demorarmos a responder
    if update_id is not None:
        if update_id in processed_updates:
            return {"ok": True}
        processed_updates.add(update_id)
        if len(processed_updates) > 1000:
            processed_updates.clear()

    # Responde 200 imediatamente e processa em segundo plano
    background_tasks.add_task(handle_update, update)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Health check (Render / UptimeRobot)
# ---------------------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "JARVIS online"}
