import asyncio
import os
import httpx
from fastapi import BackgroundTasks, FastAPI, Request

app = FastAPI()

# ---------------------------------------------------------------------------
# Configuração (variáveis de ambiente do Render)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Modelos oficiais e homologados da API do Google Gemini
MODELS_TO_TRY = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro"
]

SYSTEM_INSTRUCTION = (
    "Você é o JARVIS, um assistente pessoal inteligente que responde "
    "comandos e perguntas pelo Telegram. Responda sempre em português, "
    "de forma clara, objetiva e amigável. Ajude também com projetos de "
    "eletrônica e programação, como ESP32."
)

# ---------------------------------------------------------------------------
# Memória de conversa (em RAM)
# ---------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 20
chat_histories: dict[int, list[dict]] = {}
processed_updates: set[int] = set()


def get_gemini_contents(chat_id: int, prompt: str) -> list[dict]:
    history = chat_histories.get(chat_id, [])
    return history + [{"role": "user", "parts": [{"text": prompt}]}]


def save_to_history(chat_id: int, prompt: str, reply: str) -> None:
    history = chat_histories.setdefault(chat_id, [])
    history.append({"role": "user", "parts": [{"text": prompt}]})
    history.append({"role": "model", "parts": [{"text": reply}]})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]


# ---------------------------------------------------------------------------
# Processamento com Gemini
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
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

            for _attempt in range(2):
                try:
                    r = await client.post(url, json=payload, headers=headers)
                except Exception as e:
                    print(f"JARVIS: {model} erro de rede: {repr(e)}")
                    last_status = "timeout"
                    await asyncio.sleep(1.0)
                    continue

                print(f"JARVIS: {model} -> HTTP {r.status_code}")

                if r.status_code == 200:
                    try:
                        parts = r.json()["candidates"][0]["content"]["parts"]
                        reply = "".join(p.get("text", "") for p in parts).strip()
                    except Exception:
                        break

                    if not reply:
                        break

                    save_to_history(chat_id, prompt, reply)
                    return reply

                last_status = r.status_code

                # Em caso de sobrecarga (503) ou limitação de taxa (429), tenta o próximo modelo real
                if r.status_code in (429, 500, 503):
                    await asyncio.sleep(1.0)
                    continue

                break

    return (
        f"⚠️ Todos os modelos falharam (último erro: {last_status}). "
        "Tente de novo em instantes."
    )


# ---------------------------------------------------------------------------
# Integração com o Telegram
# ---------------------------------------------------------------------------
async def send_telegram_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [""]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            try:
                await client.post(url, json={"chat_id": chat_id, "text": chunk})
            except Exception as e:
                print(f"TELEGRAM: falha ao enviar: {repr(e)}")


async def handle_update(update: dict) -> None:
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

        if user_message.strip() in ["/limpar", "/reset"]:
            chat_histories.pop(chat_id, None)
            await send_telegram_message(chat_id, "Memória da conversa apagada. ✅")
            return

        reply = await process_with_jarvis(chat_id, user_message)
        await send_telegram_message(chat_id, reply)

    except Exception as e:
        print(f"ERRO em handle_update: {repr(e)}")


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    update = await request.json()
    update_id = update.get("update_id")

    if update_id is not None:
        if update_id in processed_updates:
            return {"ok": True}
        processed_updates.add(update_id)
        if len(processed_updates) > 1000:
            processed_updates.clear()

    background_tasks.add_task(handle_update, update)
    return {"ok": True}


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "JARVIS online"}
