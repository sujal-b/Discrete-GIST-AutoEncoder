# dge_indexer.py - RESIDUAL BEAM SEARCH
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from train_dge_final import HierarchicalDGE

class DGEIndex:
    def __init__(self, gist_codes_file="gist_codes.jsonl", checkpoint="checkpoints/dge_epoch_14.pt"):
        # Storage: keys are TUPLES (coarse, fine) -> string "12_45"
        self.clusters = {} 
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 1. Load Model
        print("⏳ Loading DGE Model (Residual VQ)...")
        self.dge = HierarchicalDGE().cuda().eval()
        # Handle state dict mismatch safely (if upgrading from old model)
        try:
            self.dge.load_state_dict(torch.load(checkpoint), strict=False)
            print("✅ Model weights loaded.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load weights fully: {e}")
        
        # 2. Load Index
        print("⏳ Loading Residual Index into Memory...")
        count = 0
        try:
            with open(gist_codes_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    # Support both old format (int) and new format (list)
                    code = entry['gist_code'] 
                    
                    # Convert list [c, f] to tuple key (c, f)
                    if isinstance(code, list):
                        key = tuple(code)
                    else:
                        # Fallback for old index (single layer)
                        key = (code, 0)
                    
                    if key not in self.clusters:
                        self.clusters[key] = []
                    
                    self.clusters[key].append(entry)
                    count += 1
            print(f"✅ Index Ready. {len(self.clusters)} active residual clusters, {count} documents.")
            
        except FileNotFoundError:
            print(f"❌ Error: {gist_codes_file} not found. Run build_gist_codes.py first.")

    def search(self, query_text, beam_width=3):
        print(f"\n🔎 Processing Query: '{query_text}'")
        
        # 1. Encode Query
        q_embed = self.encoder.encode(query_text, convert_to_tensor=True).cuda().unsqueeze(0)
        
        # 2. Neural Beam Search (Coarse -> Fine)
        with torch.no_grad():
            # Pass through Encoder
            z = self.dge.encoder(q_embed)
            
            # --- LEVEL 1: COARSE ---
            coarse_vq = self.dge.vq.layers[0]
            # Calculate distance to all 256 coarse centers
            # z shape: (1, 128), weight shape: (256, 128)
            # Normalize for spherical distance
            z_norm = torch.nn.functional.normalize(z, p=2, dim=1)
            c_weight_norm = torch.nn.functional.normalize(coarse_vq.embedding.weight, p=2, dim=1)
            
            # Dot product similarity (higher is better/closer)
            coarse_sims = torch.matmul(z_norm, c_weight_norm.t())
            # Get Top-K Coarse
            top_coarse = torch.topk(coarse_sims, k=beam_width, dim=1).indices[0]
            
            candidates = []
            
            print(f"   ↳ 📡 Exploring Coarse Clusters: {top_coarse.tolist()}")
            
            # --- LEVEL 2: FINE (Residual) ---
            for c_idx in top_coarse:
                c_idx_int = c_idx.item()
                
                # Calculate Residual: z - coarse_vector
                # We use raw vectors for subtraction (not normalized)
                residual = z - coarse_vq.embedding.weight[c_idx_int].unsqueeze(0)
                
                fine_vq = self.dge.vq.layers[1]
                
                # Search Fine Centroids
                r_norm = torch.nn.functional.normalize(residual, p=2, dim=1)
                f_weight_norm = torch.nn.functional.normalize(fine_vq.embedding.weight, p=2, dim=1)
                
                fine_sims = torch.matmul(r_norm, f_weight_norm.t())
                top_fine = torch.topk(fine_sims, k=beam_width, dim=1).indices[0]
                
                # Collect docs
                for f_idx in top_fine:
                    f_idx_int = f_idx.item()
                    bucket_key = (c_idx_int, f_idx_int)
                    
                    found_docs = self.clusters.get(bucket_key, [])
                    if found_docs:
                        candidates.extend(found_docs)
                        # Optional: Print detailed hit info
                        # print(f"      Found {len(found_docs)} docs in bucket {bucket_key}")

        print(f"   ↳ 📥 Retrieved {len(candidates)} candidates from {beam_width*beam_width} residual paths.")
        
        if not candidates:
            return []
            
        # 4. Rerank (Cosine)
        cand_texts = [c['text'] for c in candidates]
        cand_embeds = self.encoder.encode(cand_texts)
        q_embed_np = q_embed.cpu().numpy()
        
        sims = cosine_similarity(q_embed_np, cand_embeds)[0]
        
        # Top 3 Final
        top_k_final = 3
        top_idxs = np.argsort(sims)[-top_k_final:][::-1]
        
        results = []
        print("   ↳ 🏆 Final Reranked Results:")
        for rank, i in enumerate(top_idxs):
            doc = candidates[i]
            score = sims[i]
            print(f"      {rank+1}. [{score:.4f}] {doc['text'][:60]}...")
            results.append((doc['doc_id'], doc['text']))
            
        return results

# Self-test block
if __name__ == "__main__":
    idx = DGEIndex()
    idx.search("List the appearance of David Prowse in Star Wars movies.")