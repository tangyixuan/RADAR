# RADAR: Retrieval-Grounded Multi-Agent Debate for Claim Verification

A claim verification framework that integrates document retrieval, role-based multi-agent debate, and adaptive judging. Agents with complementary perspectives reason over shared evidence through structured debate to identify misleading claims caused by missing or selectively presented context. The system supports multiple debate configurations, logs intermediate reasoning steps, and evaluates predictions against ground-truth labels.

```
.
├── README.md
├── requirements.txt
├── main.py                  # Entry point for inference
├── data/                    # Input/output data
├── chroma/                  # Scripts for building/querying ChromaDB
├── agents/                  # Dialogue agents and judges
├── eval/                    # Evaluation utilities
├── model/                   # Local model weights
└── prompts/                 # Prompt templates
```

## 1. Prerequisites

Create and activate environment:

```bash
conda create -n mad_debate python=3.13.2 -y
conda activate mad_debate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Download Local Models

Two local models are supported out of the box. Download them with the Hugging Face CLI (requires accepting each model's license on the HF hub).

```bash
pip install huggingface-hub
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./model/llama3-8b-instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./model/qwen2.5-7b-instruct
```

If you plan to call the OpenAI API (`--model gpt`), supply `--api_key <key>` when running `main.py`.

## 4. Prepare the Evidence Store (ChromaDB)

1. **Populate the vector store**
   ```bash
   cd chroma
   python chroma_add.py
   ```
   - Reads `data/original_data.json`
   - Deduplicates evidence sentences
   - Indexes vectors in ChromaDB (persisted in `chroma/`)

2. **Retrieve evidence for each claim** *(optional)*
   ```bash
   python chroma_query.py
   ```
   - Performs vector similarity search with the raw claim text
   - Writes `data/retrieved_evidence_bgebase.json` (top-20 evidence list per claim)

   > Skip this step if you just need the default retrieval output. The repo already ships with `data/retrieved_evidence_bgebase.json`, and the full-evidence variant is in `data/full_evidence.json`.

## 5. Running Inference

`main.py` exposes six modes after recent pruning:

| Mode                          | Description |
|------------------------------|-------------|
| `single`                     | Single-agent verifier baseline |
| `multi`                      | Standard pro vs. con debate with a final judge |
| `multi_people`               | Politician vs. scientist debate (opening, rebuttal, closing) |
| `multi_people_continue_check`| Adds a continuation judge that can stop the debate early |
| `multi_people_round_judges`  | Introduces a judge after each round for fine-grained supervision |
| `multi_people_hybrid_adaptive` | Hybrid of round judges and continuation judge with adaptive thresholds `tau_s` and `tau_v` |

Common flags:

- `--model {llama,qwen,gpt}`
- `--model_path <local_model_dir>` for llama/qwen (defaults shown in `main.py`)
- `--input_file data/<your_input>.json`
- `--tau_s` and `--tau_v` (only used by the hybrid adaptive mode)

### Example: run the base multi-agent people debate

```bash
python main.py \
  --mode multi_people \
  --model qwen \
  --model_path ./model/qwen2.5-7b-instruct \
  --input_file data/full_evidence.json
```

### Example: hybrid adaptive mode with custom thresholds

```bash
python main.py \
  --mode multi_people_hybrid_adaptive \
  --tau_s -0.15 \
  --tau_v 0.7 \
  --model llama \
  --model_path ./model/llama3-8b-instruct \
  --input_file data/retrieved_evidence_bgebase.json
```

Outputs are written to `data/<input_basename>_answer_map_<mode>_<model>.json`.

## 7. Parameter Tuning Workflow

Use `parameter_tuning/grid_search_thresholds.py` to sweep thresholds for the hybrid pipeline:

1. Collect STOP/CONTINUE probabilities via `main.py --mode multi_people_continue_check` (produces the `--stop_file`).
2. Collect per-round verdict logs via `main.py --mode multi_people_round_judges` (produces the `--label_file`).
3. Supply ground-truth labels in `--gt_file` (e.g., `data/GT_test_all.json`).

```bash
python parameter_tuning/grid_search_thresholds.py \
  --stop_file data/retrieved_evidence_bgebase_answer_map_multi_people_continue_check_llama.json \
  --label_file data/retrieved_evidence_bgebase_answer_map_multi_people_round_judges_llama.json \
  --gt_file data/GT_test_all.json
```

The script prints the best `(tau_s, tau_v)` pair based on accuracy/F1.

## 8. Evaluation

Once answer maps are produced, load them in `eval/eval.py` to compute agreement with ground truth. Typical usage:

```bash
python eval/eval.py \
  --prediction data/my_claims_answer_map_multi_people_qwen.json
```

Adjust arguments based on the metric you need (accuracy, macro F1, etc.). Inspect `eval/eval.py` for additional options.

