from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from agent.agent import setup_agent

app = FastAPI()

@app.get("/")
async def get():
    with open("index.html", encoding="utf8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chatbot = setup_agent()
    while True:
        user_query = await websocket.receive_text()
        await websocket.send_json({
            "role": "user",
            "message": user_query
        })
        completion = chatbot.chat(user_query)  # Obtenemos la respuesta
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                new_content = chunk.choices[0].delta.content
                response += new_content
                await websocket.send_json({
                    "role": "agent",
                    "message": new_content
                })
        chatbot.save_new_message(user_query, response)
        chatbot.clear_history()

