import os
import time
import uuid
import json

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException

from loader import Loader

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


INDEX_NAME = "doc-store"
NAMESPACE = "knowledge-base"
BATCH_SIZE = 90


def check_index(index_name: str) -> None:

    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "page_content"}
            },
        )

def clean_meta(meta: dict) -> dict:

    clean = {}
    for k, v in meta.items():
        if v is None or k == "image_base64":
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list) and all(isinstance(x , str) for x in v):
            clean[k] = v
    return clean

def create_records(chunks):

    records = []
    for i, chunk in enumerate(chunks, 1):
        text = (chunk.page_content or "").strip()
        meta = clean_meta(dict(chunk.metadata))

        if not text:
            text = (meta.get("caption") or "").strip()
        if not text:
            continue
        records.append({
            "id": f"{chunk.metadata["filename"]}-{i}-{uuid.uuid4().hex[:8]}",
            "page_content": text,
            **meta,
        })

    return records

def audit(records, limit=40960):
    for i, r in enumerate(records):
        size = len(json.dumps(r).encode("utf-8"))
        if size > limit:
            print(f"record {i} ({r['id']}): {size} bytes")
            for k, v in sorted(r.items(), key=lambda kv: -len(json.dumps(kv[1]).encode())):
                print(f"   {k}: {len(json.dumps(v).encode())} bytes")
            break

def insert_records(index, records: list[dict]) -> None:

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i: i+BATCH_SIZE]

        for attempt in range(3):
            try:
                index.upsert_records(
                    namespace=NAMESPACE,
                    records=batch,
                )
                print(f"Upserted batch {i} to {i+len(batch)}")
                break

            except PineconeApiException as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"Retry {attempt+1} after {wait}s: {e}")
                time.sleep(wait)

def ingest() -> None:

    chunks = Loader.loading_pipeline()
    records = create_records(chunks)

    json.dumps(records)

    audit(records)

    index = pc.Index(INDEX_NAME)
    insert_records(index, records)

if __name__ == "__main__":
    check_index(INDEX_NAME)
    ingest()