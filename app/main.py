from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(
    title="ActionGraph",
    description="LLM-powered Meeting Intelligence Platform",
    version="0.1.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "application": "ActionGraph",
        "status": "running"
    }