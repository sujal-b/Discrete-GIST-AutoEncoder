import torch
import json
from tqdm import tqdm
from train_dge_final import HierarchicalDGE, ChunkDataset, efficient_collate_fn
from torch.utils.data import DataLoader
import os

# Update to match your actual last epoch 
CHECKPOINT = "checkpoints/dge_epoch_14.pt"
DATA_FILE = "chunks.jsonl"
OUTPUT_FILE = "gist_codes.jsonl"

if not os.path.exists(CHECKPOINT):
    print(f"⚠️ Checkpoint {CHECKPOINT} not found. Please verify the filename.")
    exit()

# 1. Load Model
model = HierarchicalDGE().cuda()
state_dict = torch.load(CHECKPOINT)
model.load_state_dict(state_dict, strict=False)
model.eval()

# 2. Load Source Lines (to preserve ID/Text mapping)
print("⏳ Reading source file...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    source_lines = [json.loads(line) for line in f]

# 3. Create Loader
dataset = ChunkDataset(DATA_FILE)
loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=efficient_collate_fn)

print("🚀 Building Index (Residual Codes)...")
results = []
line_idx = 0

with torch.no_grad():
    for batch_embeds in tqdm(loader):
        batch_embeds = batch_embeds.cuda()
        
        # Get codes: shape (Batch, 2)
        _, _, indices = model(batch_embeds)
        codes = indices.cpu().numpy() # Shape (Batch, 2)
        
        # Match with source lines
        for i in range(codes.shape[0]):
            code_pair = codes[i].tolist() # [coarse_int, fine_int]
            source = source_lines[line_idx]
            
            entry = {
                "doc_id": source["doc_id"],
                "chunk_id": source["chunk_id"],
                "text": source["text"],
                "gist_code": code_pair # Now saving the pair!
            }
            results.append(entry)
            line_idx += 1

# 4. Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"✅ Indexed {len(results)} chunks to {OUTPUT_FILE}.")
