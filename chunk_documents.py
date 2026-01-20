# chunk_documents_nq.py - FIXED for Natural Questions (JSON Array)
import json
import os

INPUT = "gist_data.json"  # Pointing to your new big dataset
OUTPUT = "chunks.jsonl"   # The standard output for the pipeline
CHUNK_SIZE_WORDS = 180
OVERLAP = 30

def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += (chunk_size - overlap)
    return chunks

print(f"⏳ Loading {INPUT} (JSON Array)...")

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    # 1. Load the huge JSON array at once (Standard JSON)
    data = json.load(fin)
    print(f"✅ Loaded {len(data)} documents. Chunking...")

    count = 0
    for doc in data:
        # 2. Extract 'context' instead of 'text'
        # Fallback to empty string if missing
        doc_id = doc.get("id", f"doc_{count}")
        text = doc.get("context", "") 
        
        if not text: continue # Skip empty docs

        chunks = chunk_text(text)
        
        for cid, c in enumerate(chunks):
            # 3. Write as standardized JSONL for train_dge_v2.py
            out = {
                "doc_id": str(doc_id),
                "chunk_id": cid,
                "text": c  # Normalized to 'text' for the pipeline
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        
        count += 1
        if count % 1000 == 0:
            print(f"   Processed {count} docs...")

print(f"🎉 Created {OUTPUT} from {len(data)} source docs.")