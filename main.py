import asyncio

MODELS_TO_TRY = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

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

            for attempt in range(2):
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
                        reply = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    except Exception:
                        print(f"JARVIS: resposta inesperada: {r.text[:1000]}")
                        break
                    save_to_history(chat_id, prompt, reply)
                    return reply

                last_status = r.status_code
                print(f"JARVIS: {r.text[:500]}")

                if r.status_code in (503, 429, 500):
                    await asyncio.sleep(1.5)
                    continue
                break

    return f"⚠️ Todos os modelos falharam (último erro: {last_status}). Tente de novo em instantes."
