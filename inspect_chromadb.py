# inspect_chromadb.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("knn-demo")
results = collection.get(limit=2)  # get first 2 chunks
print("IDs:", results['ids'])
print("Metadatas:", results['metadatas'])
print("Documents (preview):", [d[:100] for d in results['documents']])