from fastapi import FastAPI
from langserve import add_routes

from agent import agent

app = FastAPI(
    title="Internet History RAG Agent",
    version="1.0",
)

add_routes(
    app,
    agent,
    path="/agent",
    playground_type="default",
)

@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Internet History Agent",
    }
