# mock_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json, asyncio, time

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    model = body.get("model")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

    if stream:
        async def gen():
            chunks = ["Hi", ", ", "this ", "is ", "streaming", "."]
            for i, c in enumerate(chunks):
                payload = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": c},
                        "finish_reason": None if i < len(chunks) - 1 else "stop"
                    }]
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.05)
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return JSONResponse({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"Echo: {last_user}"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
#
# @app.post("/v1/embeddings")
# async def embeddings(req: Request):
#     body = await req.json()
#     inputs = body.get("input", [])
#     vec = [0.1, 0.2, 0.3, 0.4]  # fixed-length for deterministic tests
#     data = [{"object": "embedding", "index": i, "embedding": vec} for i, _ in enumerate(inputs)]
#     return JSONResponse({
#         "object": "list",
#         "data": data,
#         "model": body.get("model"),
#         "usage": {"prompt_tokens": 0, "total_tokens": 0}
#     })