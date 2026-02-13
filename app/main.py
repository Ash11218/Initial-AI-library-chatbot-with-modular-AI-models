from fastapi import FastAPI
from app.router import router

app = FastAPI(title="AI Library Chatbot")
app.include_router(router)
