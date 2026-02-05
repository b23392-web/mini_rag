import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from groq import Groq

# ---------- ENV CHECK ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

# ---------- EMBEDDINGS ----------
print("Loading Nomic embedding model...")
embed_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True
)
print("Embedding model loaded")

# ---------- QDRANT ----------
qdrant = QdrantClient(location=":memory:")
COLLECTION = "docs"

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=embed_model.get_sentence_embedding_dimension(),
        distance=Distance.COSINE
    )
)

# ---------- GROQ ----------
groq = Groq(api_key=GROQ_API_KEY)

# ---------- HELPERS ----------
def embed(text: str):
    return embed_model.encode(text, normalize_embeddings=True).tolist()

def ingest(text: str):
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=0,
                vector=embed(text),
                payload={"text": text}
            )
        ]
    )

def ask(question: str):
    results = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[],
        query=embed(question),
        limit=1,
        with_payload=True,
    )

    if not results.points:
        return "No context found."

    context = results.points[0].payload["text"]

    prompt = f"""
Answer ONLY using the context below.
Use citation [1].

Context:
[1] {context}

Question:
{question}
"""

    completion = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return completion.choices[0].message.content
