from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag import ingest, ask

app = FastAPI()

# ---------- LOAD DOCUMENT ON START ----------
DOCUMENT = """Retrieval-Augmented Generation (RAG) is a technique that improves the accuracy
and reliability of large language models by combining them with external
knowledge sources. Instead of relying only on the model’s internal training
data, RAG systems first retrieve relevant information from a document store
using vector similarity search. The retrieved content is then provided as
context to the language model before generating an answer.

A typical RAG pipeline consists of four main steps. First, documents are
split into smaller chunks and converted into numerical vector embeddings.
Second, these embeddings are stored in a vector database such as Qdrant.
Third, when a user asks a question, the query is embedded and compared
against stored vectors to find the most relevant chunks. Finally, the
language model generates a response using both the retrieved context and
the user’s question.

RAG systems are widely used in chatbots, search engines, enterprise knowledge
bases, and question-answering applications where factual correctness is
important.
"""
ingest(DOCUMENT)
print("Document ingested")

# ---------- FRONTEND ----------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

# ---------- API ----------
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_api(q: Query):
    print("RECEIVED QUESTION:", q.question)   # ← THIS WILL APPEAR IN TERMINAL
    answer = ask(q.question)
    return {"answer": answer}
