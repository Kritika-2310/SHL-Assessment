from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent.agent import run_agent

app = FastAPI(title="SHL Assessment Agent")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name:      str
    url:       str
    test_type: str


class ChatResponse(BaseModel):
    reply:               str
    recommendations:     list[Recommendation]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"Invalid role '{msg.role}'.")

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from 'user'.")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = run_agent(messages)

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=[
            Recommendation(
                name=r.get("name", ""),
                url=r.get("url", ""),
                test_type=r.get("test_type", ""),
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )
