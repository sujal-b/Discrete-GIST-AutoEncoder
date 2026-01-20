# run_dge_rag.py - UPDATED FOR RESIDUAL BEAM SEARCH
from dge_indexer import DGEIndex
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Load the Index
print("🧠 Loading DGE-RAG Index...")
# Ensure this file exists (created by build_gist_codes.py)
index = DGEIndex("gist_codes.jsonl") 

# 2. Load the Model
print("⏳ Loading TinyLlama Chat...")
token = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(token)
model = AutoModelForCausalLM.from_pretrained(
    token, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def format_tinyllama_prompt(question, contexts):
    """
    Formats the input into the strict TinyLlama chat template.
    """
    context_block = ""
    # contexts is a list of (doc_id, text) tuples
    for i, (doc_id, text) in enumerate(contexts):
        context_block += f"Source [{i+1}]: {text}\n\n"
        
    prompt = f"""<|system|>
You are a helpful AI assistant. Read the provided sources and answer the user's question naturally in your own words.
Summarize the answer clearly & structurally shape according to questions. Do not just copy-paste.
If the answer is not contained within the sources, respond with "I don't know".
</s>
<|user|>
Context Sources:
{context_block}

Question: {question}
</s>
<|assistant|>
"""
    return prompt

def answer(question):
    # 1. Search (Retrieval)
    # UPDATED: Use beam_width=3 to match your dge_indexer.py API
    # This will search 3 coarse clusters -> 3 fine clusters each (9 paths total)
    top_docs = index.search(question, beam_width=3)
    
    if not top_docs:
        return "I could not find any relevant documents in the index."

    # 2. Create Proper Chat Prompt
    prompt = format_tinyllama_prompt(question, top_docs)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 3. Generate (With Safety Settings)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,    # Allow enough space for a full answer
        do_sample=True,        # Essential for chat flow
        temperature=0.4,       # Low temperature = focused on facts
        top_p=0.9,             # Nucleus sampling
        repetition_penalty=1.1, # Prevents looping
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 4. Decode ONLY the answer (Slice off the prompt)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Interactive Loop
if __name__ == "__main__":
    print("\n✅ System Ready. Type 'exit' to quit.\n")
    while True:
        q = input("🧠 Q> ")
        if q.lower() in ["exit", "quit"]: break
        
        try:
            print(f"🔎 Thinking...")
            res = answer(q)
            print(f"🤖 A> {res}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error: {e}")