import os
import pickle
import re
import lancedb 
import pyarrow as pa 
import gc
import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor
from config import *

MAX_BATCH_SIZE = 5000
FLUSH_EVERY_N_FILES = 20


def update_db_registry(db_registry, file):
    db_registry.add(file)
    with open(db_registry_file, 'w') as registry_f:
        json.dump(list(db_registry), registry_f)

# Cartella dove LanceDB salverà il DATABASE VERO E PROPRIO
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def connect_to_db():
    # MODIFICATO: Connessione a LanceDB e creazione Tabella con Schema
    db = lancedb.connect(VECTOR_DB_PATH)
    
    # LanceDB richiede di sapere in anticipo come sono fatti i dati
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 1024)), # 1024 è la dimensione di bge-m3
        pa.field("id", pa.string())
    ])
    
    if "wiki_rag_collection" not in db.table_names():
        collection = db.create_table("wiki_rag_collection", schema=schema)
    else:
        collection = db.open_table("wiki_rag_collection")
        
    return db, collection

# Connessione iniziale
db, collection = connect_to_db() # MODIFICATO: rinominato client in db

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

files_processed_since_flush = 0 # AGGIUNTO: mancava l'inizializzazione nel codice originale

with tqdm.tqdm(total=(len(files)-len(db_registry)), desc="Popolamento Database", unit="file") as pbar:
  for file in files:

    if file in db_registry:
      #print(f"File già inserito nel database: {file}")
      continue

    current_id = extract_number(file)

    with open(os.path.join(EMBEDDINGS_PATH, file), 'rb') as embedding_f:
        chunk_embeddings = pickle.load(embedding_f)

    chunk_size = chunk_embeddings.shape[0]

    # LanceDB/Chroma richiedono che gli ID siano stringhe (es. "id_0", "id_1")
    ids_for_chunk = [str(i) for i in range(current_id, current_id + chunk_size)]

    for i in range(0, chunk_size, MAX_BATCH_SIZE):
        # Calcoliamo la fine del sotto-batch
        end_idx = min(i + MAX_BATCH_SIZE, chunk_size)

        # Estraiamo la fetta (slice) di embedding e convertiamo in float32
        sub_batch_embeddings = chunk_embeddings[i:end_idx].astype('float32').tolist()
        sub_batch_ids = ids_for_chunk[i:end_idx]

        # MODIFICATO: LanceDB accetta una lista di dizionari [{vector: [...], id: "..."}]
        data_to_insert = [
            {"vector": vec, "id": id_str} 
            for vec, id_str in zip(sub_batch_embeddings, sub_batch_ids)
        ]

        # Inserimento del sotto-batch
        collection.add(data_to_insert)

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
    if 'data_to_insert' in locals(): del data_to_insert # Pulizia extra
    gc.collect()

    files_processed_since_flush += 1

    # --- IL TRUCCO MAGICO PER LA RAM ---
    if files_processed_since_flush >= FLUSH_EVERY_N_FILES:
        # 1. Distruggiamo gli oggetti del Database
        del collection
        del db # MODIFICATO: si chiamava client
        
        # 2. Forziamo il Garbage Collector
        gc.collect()
        
        # 3. Diamo tempo al sistema operativo di scrivere su SSD e pulire la RAM
        time.sleep(2) 
        
        # 4. Riconnettiamoci
        db, collection = connect_to_db() # MODIFICATO
        files_processed_since_flush = 0


# MODIFICATO: Creazione finale dell'indice ottimizzato per le ricerche veloci
print("\nCreazione indice spaziale IVF-PQ su disco (richiederà qualche minuto)...")
collection.create_index(metric="cosine", num_partitions=256, num_sub_vectors=64)

print("Database LanceDB popolato e indicizzato con successo!")