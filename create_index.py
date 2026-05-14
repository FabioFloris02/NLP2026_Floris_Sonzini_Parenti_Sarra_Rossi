import os
import pickle
import re
import chromadb
import gc
import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor

MAX_BATCH_SIZE = 5000
EMBEDDINGS_PATH = "../Data/Embeddings"
VECTOR_DB_PATH = "../Data/db_local"
FLUSH_EVERY_N_FILES = 20


def update_db_registry(db_registry, file):
    db_registry.add(file)
    with open(db_registry_file, 'w') as registry_f:
        json.dump(list(db_registry), registry_f)



# Cartella dove ChromaDB salverà il DATABASE VERO E PROPRIO
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def connect_to_db():
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection(
        name="wiki_rag_collection",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

# Connessione iniziale
client, collection = connect_to_db()

# Create or open db_registry.json
db_registry_file = os.path.join(VECTOR_DB_PATH, "db_registry.json")

if os.path.exists(db_registry_file):
    with open(db_registry_file, 'r') as registry_f:
            db_registry = set(json.load(registry_f))
else:
    db_registry = set()

# Take embeddings .plk files from directory and sorts for chunk number
files = [f for f in os.listdir(EMBEDDINGS_PATH) if f.endswith('.pkl')]

def extract_number(filename):
    match = re.search(r'chunk_(\d+)_', filename)
    return int(match.group(1)) if match else 0

files.sort(key=extract_number)

# 3. Inserimento Incrementale nel Database


with tqdm.tqdm(total=(len(files)-len(db_registry)), desc="Popolamento Database", unit="file") as pbar:
  for file in files:

    if file in db_registry:
      #print(f"File già inserito nel database: {file}")
      continue

    current_id = extract_number(file)

    with open(os.path.join(EMBEDDINGS_PATH, file), 'rb') as embedding_f:
        chunk_embeddings = pickle.load(embedding_f)

    chunk_size = chunk_embeddings.shape[0]

    # ChromaDB richiede che gli ID siano stringhe (es. "id_0", "id_1")
    ids_for_chunk = [str(i) for i in range(current_id, current_id + chunk_size)]

    for i in range(0, chunk_size, MAX_BATCH_SIZE):
        # Calcoliamo la fine del sotto-batch
        end_idx = min(i + MAX_BATCH_SIZE, chunk_size)

        # Estraiamo la fetta (slice) di embedding e ID
        sub_batch_embeddings = chunk_embeddings[i:end_idx].tolist()
        sub_batch_ids = ids_for_chunk[i:end_idx]

        # Inserimento del sotto-batch
        collection.add(
            embeddings=sub_batch_embeddings,
            ids=sub_batch_ids
        )

    pbar.set_postfix({"ultimo_file": file})
    pbar.update(1)

    db_registry.add(file)
    with open(db_registry_file, 'w') as registry_f:
        json.dump(list(db_registry), registry_f)



    # Pulizia RAM di Python
    del chunk_embeddings
    del ids_for_chunk
    del sub_batch_embeddings
    del sub_batch_ids
    gc.collect()

    files_processed_since_flush += 1

    # --- IL TRUCCO MAGICO PER LA RAM ---
    if files_processed_since_flush >= FLUSH_EVERY_N_FILES:
        # 1. Distruggiamo gli oggetti ChromaDB
        del collection
        del client
        
        # 2. Forziamo il Garbage Collector
        gc.collect()
        
        # 3. Diamo tempo al sistema operativo di scrivere su SSD e pulire la RAM
        time.sleep(2) 
        
        # 4. Riconnettiamoci (ChromaDB leggerà i dati freschi dal disco)
        client, collection = connect_to_db()
        files_processed_since_flush = 0


print("Database Chroma popolato con successo su Google Drive!")