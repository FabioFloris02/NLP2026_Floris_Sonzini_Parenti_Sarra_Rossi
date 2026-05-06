# Import dataset from parquet
from datasets import load_dataset

def load_dataset(parquet_path):
    ds = load_dataset("parquet", data_files=parquet_path, split="train")

    print(ds[0])


    corpus_docs = list(ds['description'])
    corpus_ids = list(range(len(corpus_docs)))
    corpus_docs[:6]