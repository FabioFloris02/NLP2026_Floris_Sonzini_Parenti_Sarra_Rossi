# Durante la query RAG:
import lancedb 
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from local_deployment.config import *
from datasets import load_dataset
import os


bi_enc = SentenceTransformer('BAAI/bge-m3', model_kwargs={"torch_dtype": torch.float16},)

db = lancedb.connect(VECTOR_DB_PATH)
collection = db.open_table("wiki_rag_collection")

query="chi è il mago forest?"
query_vector = bi_enc.encode([query]).tolist()[0]

reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024) 

# Ricerca (equivale a collection.query() di Chroma)
risultati = collection.search(query_vector).limit(30).to_pandas()

if (os.path.exists(PARQUET_PATH)):
  print("Ok")

ds = load_dataset("parquet", data_files=PARQUET_PATH, split="train")

# `risultati` is a pandas DataFrame; iterate its `id` column (row ids)
retrieved_docs = []
for i in risultati['id'].tolist():
  idx = int(i)
  batch_docs = ds[idx]['content']
  #print(batch_docs+"\n\n")
  retrieved_docs.append(batch_docs)

couples = [[query, doc] for doc in retrieved_docs]
scores= reranker.predict(couples)

docs_with_score = list(zip(scores, retrieved_docs))
docs_with_score.sort(key=lambda x:x[0], reverse=True)

best_doc=docs_with_score[0][1]

print(f"IL VINCITORE ASSOLUTO È:\n{best_doc}")
