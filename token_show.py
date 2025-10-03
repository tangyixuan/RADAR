from transformers import AutoTokenizer

MODEL = "/home/y/yirui-z/mad_formal/model/llama3-8b-instruct"
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

def show(s):
    ids = tok(s).input_ids
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"text={repr(s)}")
    print("num_tokens=", len(ids))
    print("tokens=", pieces)
    print()

show("production decisions.\n\nVERDICT: HALF-TRUE")
show("production decisions.\n\nVERDICT: TRUE")
show("production decisions.\n\nVERDICT: FALSE")