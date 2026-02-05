# mini_rag
This project is a small RAG (Retrieval-Augmented Generation) system built using FastAPI.
Text is split into chunks, converted into embeddings using the Nomic open-source embedding model, and stored in an in-memory Qdrant vector database.

When a question is asked, the system retrieves the most relevant chunks and uses them to generate an answer with citations, instead of guessing.

The goal of this project is to demonstrate the end-to-end RAG flow in a minimal, production-minded way.
