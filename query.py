# query.py
import os
from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = "rag_data/chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

def query(question, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("syllabi")

    query_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)

    print(f"\n🔎 Question: {question}\n")
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0],
                                              results["metadatas"][0],
                                              results["distances"][0])):
        print(f"--- Result {i+1} (distance: {dist:.4f}) ---")
        print(f"Source: {meta['source']}")
        print(doc[:500] + "...\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
    else:
        query("What are the core courses for M.Sc. Physics?")
