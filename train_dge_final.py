import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
CHECKPOINT_DIR = "checkpoints"
DATA_FILE = "chunks.jsonl" 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 1. BASE VECTOR QUANTIZER ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_(0, 0.01)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))

    def init_codebook(self, data):
        print(f"⚡ Initializing {self.num_embeddings} codes...")
        data = F.normalize(data, p=2, dim=1)
        if data.size(0) < self.num_embeddings:
            # Handle small data init case
            noise = torch.randn(self.num_embeddings - data.size(0), self.embedding_dim).to(data.device)
            noise = F.normalize(noise, p=2, dim=1)
            data = torch.cat([data, noise], dim=0)
            
        self.embedding.weight.data.copy_(data[:self.num_embeddings])
        self.ema_w.data.copy_(self.embedding.weight.data)
        self.ema_cluster_size.fill_(1)

    def forward(self, inputs):
        # Flatten inputs
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Spherical Distance Calculation
        codebook = F.normalize(self.embedding.weight, p=2, dim=1)
        inputs_norm = F.normalize(flat_input, p=2, dim=1)
        distances = 2 - 2 * torch.matmul(inputs_norm, codebook.t())
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Training (EMA updates)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            )
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
            
        # Loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        quantized = inputs + (quantized - inputs).detach() # STE
        
        return quantized, e_latent_loss, encoding_indices

# --- 2. RESIDUAL VQ (The Upgrade) ---
class ResidualVQ(nn.Module):
    def __init__(self, num_quantizers=2, num_embeddings=256, dim=384):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, dim) for _ in range(num_quantizers)
        ])
    
    def forward(self, x):
        residual = x
        quantized_out = 0
        all_indices = []
        all_losses = 0
        
        for layer in self.layers:
            z_q, loss, indices = layer(residual)
            quantized_out = quantized_out + z_q
            residual = residual - z_q.detach()
            all_indices.append(indices)
            all_losses += loss
            
        # Stack codes: (Batch, 2)
        codes = torch.stack(all_indices, dim=1).squeeze(-1)
        return quantized_out, all_losses, codes
    
    def init_codebook(self, x):
        self.layers[0].init_codebook(x)
        residual = x - self.layers[0](x)[0].detach()
        self.layers[1].init_codebook(residual)

# --- 3. HIERARCHICAL MODEL ---
class HierarchicalDGE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 384 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        # 2-Layer Residual VQ
        self.vq = ResidualVQ(num_quantizers=2, num_embeddings=256, dim=128)
        
        # Decoder: 128 -> 384
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 384)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices

# --- 4. DATASET & TRAINING ---
class ChunkDataset(Dataset):
    def __init__(self, chunks_file):
        self.data = []
        print("⏳ Loading dataset into RAM...")
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    self.data.append(json.loads(line)["text"])
        except FileNotFoundError:
            print(f"❌ Error: {chunks_file} not found. Run chunk_documents.py first.")
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx] 

global_encoder = SentenceTransformer('all-MiniLM-L6-v2')
global_encoder.eval()

def efficient_collate_fn(batch_texts):
    with torch.no_grad():
        return global_encoder.encode(batch_texts, convert_to_tensor=True)

def train():
    dataset = ChunkDataset(DATA_FILE)
    if len(dataset) == 0: return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        collate_fn=efficient_collate_fn, num_workers=0)
    
    model = HierarchicalDGE().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("⚡ Bootstrapping codebook...")
    init_batch = next(iter(loader))
    init_batch = init_batch.detach().clone().cuda()
    with torch.no_grad():
        z_init = model.encoder(init_batch)
        model.vq.init_codebook(z_init)

    print(f"🚀 Starting training on {len(dataset)} chunks...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_recon_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for batch_embeds in loop:
            batch_embeds = batch_embeds.detach().clone().cuda()
            batch_embeds_norm = F.normalize(batch_embeds, p=2, dim=1)
            
            recon, vq_loss, indices = model(batch_embeds)
            
            recon_loss = F.mse_loss(recon, batch_embeds_norm)
            loss = recon_loss + 0.25 * vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            loop.set_postfix(loss=f"{recon_loss.item():.4f}")
        
        avg_loss = total_recon_loss / len(loader)
        print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f}")
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/dge_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()