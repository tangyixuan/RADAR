import json
import hashlib
from tqdm import tqdm
from chroma import ChromaClient

# Load examples from file
with open("../data/original_data.json", "r") as f:
    all_examples = json.load(f)

chroma_client = ChromaClient(vector_name="evidence_bgebase")

# Global evidence counter
global_evidence_id = 0

# Use a hash set to skip duplicate content
seen_evidence_hashes = set()

# Mapping: example_id → list of assigned evidence_id
example_to_evidence_map = {}

# Mapping: sentence_hash → evidence_id
evidence_hash_to_id = {}

# Mapping: sentence_text → evidence_id
evidence_text_to_id = {}

# Mapping: evidence_id → sentence_text
evidence_id_to_text = {}

# Mapping: example_id → claim
example_to_claim = {}

# Main insertion loop with progress bar
for example in tqdm(all_examples, desc="Processing examples"):
    example_id = example["example_id"]
    claim = example["claim"]
    evidence_list = example["evidence"]
    example_to_evidence_map[example_id] = []
    example_to_claim[example_id] = claim

    for sentence in evidence_list:
        # Normalize and hash sentence to detect duplicates
        sentence_norm = sentence.strip()
        sentence_hash = hashlib.md5(sentence_norm.encode('utf-8')).hexdigest()

        if sentence_hash in evidence_hash_to_id:
            evidence_id = evidence_hash_to_id[sentence_hash]
        else:
            evidence_id = global_evidence_id
            evidence_hash_to_id[sentence_hash] = evidence_id
            evidence_text_to_id[sentence_norm] = evidence_id
            evidence_id_to_text[evidence_id] = sentence_norm
            global_evidence_id += 1
            chroma_client.add_document(
                content=sentence,
                metadata={
                    "evidence_id": evidence_id
                }
            )
        example_to_evidence_map[example_id].append(evidence_id)

print(f"\nInserted {global_evidence_id} unique evidence sentences.")