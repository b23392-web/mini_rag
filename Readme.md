# Mini RAG Application

## Architecture
- Embeddings: Nomic (local)
- Vector DB: Qdrant (in-memory)
- Retriever: Cosine similarity
- LLM: Groq (LLaMA-3)
- Backend: FastAPI
- Frontend: HTML + JS

## Chunking
- Strategy: Fixed chunk (demo uses 1 chunk)
- Embedding prefix:
  - search_document
  - search_query

## How to Run 
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
""" please ask question related to this part
```bash
pip install -r requirement.txt
click on index

RESUME LINK
https://drive.google.com/file/d/1_8CMEr3ki1PwoQTkEaG1b07Jej7bruEV/view?usp=sharing
