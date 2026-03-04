from chroma import ChromaClient
import json
from tqdm import tqdm

# Initialize ChromaDB client with the same collection name used during insertion
chroma_client = ChromaClient(vector_name="evidence_bgebase", path="./chroma_store")

# Load evidence_id_to_text mapping
with open("../data/evidence_id_to_text.json", "r") as f:
    evidence_id_to_text = json.load(f)

# Input claim to retrieve evidence for
with open("../data/original_data.json", "r") as f:
    all_examples = json.load(f)
example_to_retrieved_evidence_map = {}
for example in tqdm(all_examples, desc="Processing examples"):
    claim = example["claim"]
    example_id = example["example_id"]
    
    # Perform vector similarity search
    results = chroma_client.query(query_text=claim, top_k=20, include=["documents", "metadatas"])
    
    evidence_ids = []
    evidence_text = []
    for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        evidence_id = meta["evidence_id"]
        evidence_ids.append(evidence_id)
        evidence_text.append(evidence_id_to_text[str(evidence_id)])
    
    example_to_retrieved_evidence_map[example_id] = {
        "claim": claim,
        "top_20_evidences_ids": evidence_ids,
        "evidence_full_text": evidence_text
    }

# Save mapping to JSON
with open("../data/retrieved_evidence_bgebase.json", "w") as f:
    json.dump(example_to_retrieved_evidence_map, f, indent=2)