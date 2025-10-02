from model.loader import load_model
from prompts.templates_people import (
    get_system_prompt,
    politician_opening_prompt,
    politician_rebuttal_prompt,
    politician_closing_prompt,
    scientist_opening_prompt,
    scientist_rebuttal_prompt,
    scientist_closing_prompt,
    judge_prompt,
    judge_prompt_1r,
    judge_prompt_2r
)
from agents.chat_template_utils import build_chat_prompt, extract_assistant_response
import torch
import math
import re
from typing import Dict

# Global model info - will be set by main.py
model_info = None


MCQ_CHOICES = ("TRUE", "FALSE", "HALF-TRUE")
MCQ_VERDICT_MAP = {
    "TRUE": "TRUE",
    "FALSE": "FALSE",
    "HALF-TRUE": "HALF-TRUE",
}

def set_model_info(info):
    """Set the global model info"""
    global model_info
    model_info = info

def extract_verdict_and_probability(response_text, logprobs=None, scores=None, tokenizer=None, generated_tokens=None):
    """Extract verdict and probability from judge response"""
    import re
    import torch
    import torch.nn.functional as F
    import math
    
    # Extract verdict - try multiple patterns
    verdict = "UNKNOWN"
    
    # Pattern 1: [VERDICT]: TRUE/FALSE/HALF-TRUE
    verdict_match = re.search(r'\[VERDICT\]:\s*(TRUE|FALSE|HALF-TRUE)', response_text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        # Pattern 2: VERDICT: TRUE/FALSE/HALF-TRUE (without brackets)
        verdict_match = re.search(r'VERDICT:\s*(TRUE|FALSE|HALF-TRUE)', response_text, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()
        else:
            # Pattern 3: Just look for TRUE/FALSE/HALF-TRUE at the end
            verdict_match = re.search(r'\b(TRUE|FALSE|HALF-TRUE)\b', response_text, re.IGNORECASE)
            if verdict_match:
                verdict = verdict_match.group(1).upper()
    
    print(f"Extracted verdict: {verdict} from text ending: ...{response_text[-100:]}")  # Debug print
    
    probability = None
    print(f"Starting probability extraction - logprobs: {logprobs is not None}, scores: {scores is not None}, tokenizer: {tokenizer is not None}, tokens: {generated_tokens is not None}")
    
    # Try to extract probability from logprobs (OpenAI)
    if logprobs is not None:
        try:
            # Find the verdict token in the logprobs
            verdict_tokens = ["TRUE", "FALSE", "HALF-TRUE"]
            for token_logprob in logprobs:
                if token_logprob.token in verdict_tokens:
                    # Convert log probability to probability
                    probability = math.exp(token_logprob.logprob)
                    print(f"Found verdict token '{token_logprob.token}' with probability: {probability}")
                    break
        except Exception as e:
            print(f"Warning: Could not extract probability from logprobs: {e}")
    
    # Try to extract probability from scores (Hugging Face) - improved method
    elif scores is not None and tokenizer is not None and generated_tokens is not None:
        try:
            print(f"Scores type: {type(scores)}, is tuple: {isinstance(scores, tuple)}")
            
            # Handle tuple of scores (from model.generate with output_scores=True)
            if isinstance(scores, tuple):
                print(f"Scores tuple length: {len(scores)}")
                
                # Get verdict token IDs for each verdict word
                verdict_token_ids = {}
                for verdict_word in ["TRUE", "FALSE", "HALF-TRUE"]:
                    tokens = tokenizer.encode(verdict_word, add_special_tokens=False)
                    verdict_token_ids[verdict_word] = tokens
                
                print(f"Verdict token IDs: {verdict_token_ids}")
                print(f"Generated tokens (first 20): {generated_tokens[:20]}")
                
                # Find verdict tokens in generated sequence and calculate their probabilities
                verdict_probs = []
                
                for i, score_tensor in enumerate(scores):
                    # score_tensor is the logits for step i (shape: [batch_size, vocab_size])
                    if i < len(generated_tokens):
                        token_id = generated_tokens[i]
                        
                        # Check if this token is part of any verdict
                        is_verdict_token = False
                        for verdict_word, token_list in verdict_token_ids.items():
                            if token_id in token_list:
                                is_verdict_token = True
                                break
                        
                        if is_verdict_token:
                            # Calculate probability for this token
                            token_probs = F.softmax(score_tensor, dim=-1)
                            token_prob = token_probs[0, token_id].item()
                            verdict_probs.append(token_prob)
                            token_text = tokenizer.decode([token_id])
                            print(f"Step {i}: Token {token_id} ('{token_text}') probability: {token_prob:.4f}")
                
                if verdict_probs:
                    # Calculate geometric mean (equivalent to exp(mean(log_probs)))
                    # This is the most theoretically sound for joint probability of multi-token sequences
                    import math
                    probability = math.exp(sum(math.log(p) for p in verdict_probs) / len(verdict_probs))
                    
                    print(f"✅ Verdict token probabilities: {[f'{p:.4f}' for p in verdict_probs]}")
                    print(f"   Final probability (geometric mean): {probability:.4f}")
                else:
                    print(f"⚠️ No verdict tokens found in generated sequence")
                    print(f"   Expected verdict: {verdict}")
                    print(f"   Looking for token IDs in: {verdict_token_ids.get(verdict, 'Not found')}")
                    print(f"   Generated tokens (first 50): {generated_tokens[:50]}")
                    
                    # Fallback: use average probability of last few tokens as confidence estimate
                    if len(scores) > 0:
                        print(f"   Attempting fallback: using last tokens' confidence")
                        last_n = min(5, len(scores))
                        last_probs = []
                        for i in range(len(scores) - last_n, len(scores)):
                            token_logits = scores[i]
                            token_probs_dist = F.softmax(token_logits, dim=-1)
                            max_prob = torch.max(token_probs_dist).item()
                            last_probs.append(max_prob)
                        if last_probs:
                            probability = sum(last_probs) / len(last_probs)
                            print(f"   Fallback probability (avg of last {last_n} tokens): {probability:.4f}")
            else:
                print(f"Scores is not a tuple, type: {type(scores)}")
                    
        except Exception as e:
            print(f"Warning: Could not extract probability from scores: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback: use maximum probability of last token as confidence proxy
    elif scores is not None:
        try:
            if len(scores) > 0:
                last_scores = scores[-1]  # Get scores for the last generated token
                probabilities = torch.softmax(last_scores, dim=-1)
                # Get the maximum probability as a proxy for confidence
                probability = float(torch.max(probabilities).item())
                print(f"Fallback: Using max probability of last token: {probability}")
        except Exception as e:
            print(f"Warning: Could not extract probability from scores (fallback): {e}")
    
    return verdict, probability

def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300, get_probabilities: bool = False):
    """Run model inference based on model type"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    if len(model_info) == 2:
        first, second = model_info
        
        # 通过检查第二个元素来区分模型类型
        if isinstance(second, str):
            # GPT model: (client, model_name)
            client, model_name = model_info
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]  
            
            if get_probabilities:
                # Use completion API with logprobs for probability estimation
                try:
                    response = client.completions.create(
                        model=model_name,
                        prompt=f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                        max_tokens=max_tokens,
                        temperature=0.7,
                        logprobs=5,
                        top_logprobs=5
                    )
                    content = response.choices[0].text.strip()
                    logprobs = response.choices[0].logprobs
                    return content, logprobs
                except Exception as e:
                    print(f"Warning: Could not get logprobs from GPT model: {e}")
                    # Fallback to regular chat completion
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip(), None
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
        
        else:
            # Local model: (tokenizer, model)
            tokenizer, model = model_info

            text, used_chat_template = build_chat_prompt(tokenizer, system_prompt, user_prompt)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            if get_probabilities:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                extracted_response = extract_assistant_response(response, used_chat_template)
                
                # Extract only the newly generated tokens (excluding input tokens)
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs.sequences[0][input_length:].tolist()
                
                print(f"Generated response length: {len(extracted_response)}")  # Debug print
                print(f"Scores available: {outputs.scores is not None if hasattr(outputs, 'scores') else 'No scores'}")
                if hasattr(outputs, 'scores') and outputs.scores is not None:
                    print(f"Scores type: {type(outputs.scores)}, length: {len(outputs.scores)}")
                    print(f"Generated tokens: {generated_tokens}")
                
                return extracted_response, (outputs.scores, tokenizer, generated_tokens)
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return extract_assistant_response(response, used_chat_template)
    
    else:
        raise ValueError("Invalid model_info format")


def _extract_reason_section(response_text: str) -> str:
    """Extract the reason section without the verdict line."""
    reason_pattern = r'((?:\[)?REASON(?:\])?:.*?)(?=\n\s*(?:\[)?VERDICT(?:\])?:)'  # capture until verdict marker (with/without brackets)
    match = re.search(reason_pattern, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        reason_section = match.group(1).strip()
    else:
        # Fallback: remove any verdict lines and keep the rest
        lines = [
            line for line in response_text.strip().splitlines()
            if not re.match(r'\s*(?:\[)?VERDICT(?:\])?:', line, re.IGNORECASE)
        ]
        reason_section = "\n".join(lines).strip()

    if not reason_section:
        return ""

    reason_section = reason_section.rstrip()

    if reason_section and not reason_section.endswith("\n"):
        reason_section += "\n"
    return reason_section


def _logprobs_to_probabilities(logprob_map: Dict[str, float]) -> Dict[str, float]:
    """Convert choice log-probabilities into a probability distribution."""
    max_logprob = max(logprob_map.values())
    exp_map = {choice: math.exp(logprob - max_logprob) for choice, logprob in logprob_map.items()}
    total = sum(exp_map.values())
    return {choice: (value / total if total > 0 else 0.0) for choice, value in exp_map.items()}


def _common_prefix_length(first: list[int], second: list[int]) -> int:
    """Return the length of the longest shared prefix between two token id lists."""
    max_len = min(len(first), len(second))
    idx = 0
    while idx < max_len and first[idx] == second[idx]:
        idx += 1
    return idx


def _compute_choice_logprobs_local(tokenizer, model, system_prompt: str, user_prompt: str, assistant_context: str) -> Dict[str, float]:
    """Compute per-choice average log-probabilities for local models (handles multi-token labels)."""
    prompt_text, _ = build_chat_prompt(tokenizer, system_prompt, user_prompt)
    base_text = prompt_text + assistant_context

    base_encoding = tokenizer([base_text], return_tensors="pt")
    base_ids = base_encoding["input_ids"][0].tolist()

    choice_logprobs: Dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for choice in MCQ_CHOICES:
            choice_text = base_text + choice
            choice_encoding = tokenizer([choice_text], return_tensors="pt")
            choice_ids = choice_encoding["input_ids"][0].tolist()

            prefix_len = _common_prefix_length(base_ids, choice_ids)

            labels = choice_encoding["input_ids"].clone()
            labels[:, :prefix_len] = -100
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

            token_count = int((labels != -100).sum().item())
            if token_count == 0:
                choice_logprobs[choice] = float("-inf")
                continue

            choice_inputs = {k: v.to(model.device) for k, v in choice_encoding.items()}
            labels = labels.to(model.device)

            model_kwargs = {
                "input_ids": choice_inputs["input_ids"],
                "labels": labels,
            }
            if "attention_mask" in choice_inputs:
                model_kwargs["attention_mask"] = choice_inputs["attention_mask"]
            if "position_ids" in choice_inputs:
                model_kwargs["position_ids"] = choice_inputs["position_ids"]

            outputs = model(**model_kwargs)

            sum_logprob = -outputs.loss.item() * token_count
            choice_logprobs[choice] = sum_logprob / token_count

    return choice_logprobs


def _compute_choice_logprobs_gpt(client, model_name: str, system_prompt: str, user_prompt: str, assistant_context: str) -> Dict[str, float]:
    """Compute next-token log-probabilities for choices using GPT-style models."""
    base_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: {assistant_context}"
    base_length = len(base_prompt)

    choice_logprobs: Dict[str, float] = {}
    for choice in MCQ_CHOICES:
        prompt_with_choice = base_prompt + choice
        response = client.completions.create(
            model=model_name,
            prompt=prompt_with_choice,
            max_tokens=0,
            temperature=0,
            logprobs=1,
            echo=True
        )
        logprob_data = response.choices[0].logprobs
        if logprob_data is None or logprob_data.tokens is None:
            raise RuntimeError("Logprob data unavailable for GPT MCQ scoring.")

        tokens = logprob_data.tokens
        token_logprobs = logprob_data.token_logprobs
        text_offsets = logprob_data.text_offset

        recorded = False
        for delta in (0, 1, 2):
            first_idx = None
            for idx, offset in enumerate(text_offsets):
                if offset is not None and offset >= base_length - delta:
                    first_idx = idx
                    break

            if first_idx is None:
                continue

            sum_logprob = 0.0
            token_count = 0
            for idx in range(first_idx, len(tokens)):
                offset = text_offsets[idx]
                token_logprob = token_logprobs[idx]
                if offset is None or token_logprob is None:
                    continue
                if offset < base_length - delta:
                    continue
                sum_logprob += token_logprob
                token_count += 1

            if token_count > 0:
                choice_logprobs[choice] = sum_logprob / token_count
                recorded = True
                break

        if not recorded:
            choice_logprobs[choice] = float("-inf")

    return choice_logprobs


def compute_choice_logprobs(system_prompt: str, user_prompt: str, assistant_context: str) -> Dict[str, float]:
    """Dispatch next-token log-probability computation based on loaded model."""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")

    if len(model_info) != 2:
        raise ValueError("Invalid model_info format")

    first, second = model_info
    if isinstance(second, str):
        client, model_name = model_info
        return _compute_choice_logprobs_gpt(client, model_name, system_prompt, user_prompt, assistant_context)

    tokenizer, model = model_info
    return _compute_choice_logprobs_local(tokenizer, model, system_prompt, user_prompt, assistant_context)


def _compute_verdict_probability_info(user_prompt: str, assistant_context: str) -> Dict[str, object]:
    """Compute verdict token probabilities given context."""
    system_prompt = get_system_prompt("judge")
    choice_logprobs = compute_choice_logprobs(system_prompt, user_prompt, assistant_context)

    verdict_logprobs = {
        MCQ_VERDICT_MAP[choice]: logprob for choice, logprob in choice_logprobs.items()
    }
    verdict_probabilities = _logprobs_to_probabilities(verdict_logprobs)
    predicted_verdict = max(verdict_probabilities, key=verdict_probabilities.get)

    return {
        "predicted_verdict": predicted_verdict,
        "verdict_probabilities": verdict_probabilities,
        "verdict_logprobs": verdict_logprobs,
        "choice_logprobs": choice_logprobs,
    }

# === Politician Agent ===
def opening_politician(claim, evidence):
    prompt = politician_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, opponent_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence):
    prompt = politician_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence):
    prompt = scientist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, opponent_argument):
    prompt = scientist_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence):
    prompt = scientist_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

# === Round Judge Agents ===
def judge_round_1(claim, evidence, pol_open, sci_open):
    """Judge after first round (opening statements)"""
    prompt = judge_prompt_1r(claim, evidence, pol_open, sci_open)
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, get_probabilities=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
        print(f"Round 1 - Got probability data type: {type(probability_data)}")  # Debug print
        
        # Handle different probability data formats
        if probability_data is not None:
            if isinstance(probability_data, tuple) and len(probability_data) == 3:
                # New format: (scores, tokenizer, generated_tokens)
                scores, tokenizer, generated_tokens = probability_data
                verdict, probability = extract_verdict_and_probability(
                    response, scores=scores, tokenizer=tokenizer, generated_tokens=generated_tokens
                )
            elif hasattr(probability_data, '__iter__') and not isinstance(probability_data, torch.Tensor):
                # It's logprobs (OpenAI format)
                verdict, probability = extract_verdict_and_probability(response, logprobs=probability_data)
            elif isinstance(probability_data, torch.Tensor):
                # It's scores (Hugging Face format - old format)
                verdict, probability = extract_verdict_and_probability(response, scores=probability_data)
            else:
                # Unknown format, try without probability data
                verdict, probability = extract_verdict_and_probability(response)
        else:
            verdict, probability = extract_verdict_and_probability(response)
    else:
        response = result
        verdict, probability = extract_verdict_and_probability(response)
    
    return {
        "response": response,
        "verdict": verdict,
        "probability": probability
    }


def judge_round_1_mcq(claim, evidence, pol_open, sci_open):
    base_result = judge_round_1(claim, evidence, pol_open, sci_open)

    reason_section = _extract_reason_section(base_result["response"])
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "
    user_prompt = judge_prompt_1r(claim, evidence, pol_open, sci_open)
    probability_info = _compute_verdict_probability_info(user_prompt, assistant_context)

    augmented_result = dict(base_result)
    augmented_result.update({
        "reason_section": reason_section.strip(),
        "probabilistic_verdict": probability_info["predicted_verdict"],
        "verdict_probabilities": probability_info["verdict_probabilities"],
        "verdict_logprobs": probability_info["verdict_logprobs"],
        "choice_logprobs": probability_info["choice_logprobs"],
        "choice_probabilities": _logprobs_to_probabilities(probability_info["choice_logprobs"]),
    })

    return augmented_result

def judge_round_2(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut):
    """Judge after second round (opening + rebuttal statements)"""
    prompt = judge_prompt_2r(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, get_probabilities=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
        print(f"Round 2 - Got probability data type: {type(probability_data)}")  # Debug print
        
        # Handle different probability data formats
        if probability_data is not None:
            if isinstance(probability_data, tuple) and len(probability_data) == 3:
                # New format: (scores, tokenizer, generated_tokens)
                scores, tokenizer, generated_tokens = probability_data
                verdict, probability = extract_verdict_and_probability(
                    response, scores=scores, tokenizer=tokenizer, generated_tokens=generated_tokens
                )
            elif hasattr(probability_data, '__iter__') and not isinstance(probability_data, torch.Tensor):
                # It's logprobs (OpenAI format)
                verdict, probability = extract_verdict_and_probability(response, logprobs=probability_data)
            elif isinstance(probability_data, torch.Tensor):
                # It's scores (Hugging Face format - old format)
                verdict, probability = extract_verdict_and_probability(response, scores=probability_data)
            else:
                # Unknown format, try without probability data
                verdict, probability = extract_verdict_and_probability(response)
        else:
            verdict, probability = extract_verdict_and_probability(response)
    else:
        response = result
        verdict, probability = extract_verdict_and_probability(response)
    
    return {
        "response": response,
        "verdict": verdict,
        "probability": probability
    }


def judge_round_2_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut):
    base_result = judge_round_2(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    reason_section = _extract_reason_section(base_result["response"])
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "
    user_prompt = judge_prompt_2r(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    probability_info = _compute_verdict_probability_info(user_prompt, assistant_context)

    augmented_result = dict(base_result)
    augmented_result.update({
        "reason_section": reason_section.strip(),
        "probabilistic_verdict": probability_info["predicted_verdict"],
        "verdict_probabilities": probability_info["verdict_probabilities"],
        "verdict_logprobs": probability_info["verdict_logprobs"],
        "choice_logprobs": probability_info["choice_logprobs"],
        "choice_probabilities": _logprobs_to_probabilities(probability_info["choice_logprobs"]),
    })

    return augmented_result

# === Final Judge Agent ===
def judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    prompt = judge_prompt(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_close, sci_close
    )
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, get_probabilities=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
        print(f"Final - Got probability data type: {type(probability_data)}")  # Debug print
        
        # Handle different probability data formats
        if probability_data is not None:
            if isinstance(probability_data, tuple) and len(probability_data) == 3:
                # New format: (scores, tokenizer, generated_tokens)
                scores, tokenizer, generated_tokens = probability_data
                verdict, probability = extract_verdict_and_probability(
                    response, scores=scores, tokenizer=tokenizer, generated_tokens=generated_tokens
                )
            elif hasattr(probability_data, '__iter__') and not isinstance(probability_data, torch.Tensor):
                # It's logprobs (OpenAI format)
                verdict, probability = extract_verdict_and_probability(response, logprobs=probability_data)
            elif isinstance(probability_data, torch.Tensor):
                # It's scores (Hugging Face format - old format)
                verdict, probability = extract_verdict_and_probability(response, scores=probability_data)
            else:
                # Unknown format, try without probability data
                verdict, probability = extract_verdict_and_probability(response)
        else:
            verdict, probability = extract_verdict_and_probability(response)
    else:
        response = result
        verdict, probability = extract_verdict_and_probability(response)
    
    return {
        "response": response,
        "verdict": verdict,
        "probability": probability
    }


def judge_final_verdict_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    base_result = judge_final_verdict(
        claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close
    )
    reason_section = _extract_reason_section(base_result["response"])
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "
    user_prompt = judge_prompt(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_close, sci_close
    )
    probability_info = _compute_verdict_probability_info(user_prompt, assistant_context)

    augmented_result = dict(base_result)
    augmented_result.update({
        "reason_section": reason_section.strip(),
        "probabilistic_verdict": probability_info["predicted_verdict"],
        "verdict_probabilities": probability_info["verdict_probabilities"],
        "verdict_logprobs": probability_info["verdict_logprobs"],
        "choice_logprobs": probability_info["choice_logprobs"],
        "choice_probabilities": _logprobs_to_probabilities(probability_info["choice_logprobs"]),
    })

    return augmented_result

# === Main Debate Function ===
def run_multi_agent_people_round(claim, evidence):
    """Run the complete multi-agent people debate with round judges"""
    print("\n=== Running Multi-Agent People Debate with Round Judges ===")
    
    # Round 1: Opening statements
    print("Round 1: Opening statements...")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    
    # Round 1 Judge
    print("Round 1 Judge evaluation...")
    round_1_judge = judge_round_1(claim, evidence, pol_open, sci_open)
    
    # Round 2: Rebuttal statements
    print("Round 2: Rebuttal statements...")
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
    
    # Round 2 Judge
    print("Round 2 Judge evaluation...")
    round_2_judge = judge_round_2(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    
    # Round 3: Closing statements
    print("Round 3: Closing statements...")
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)
    
    # Final Judge
    print("Final Judge evaluation...")
    final_judge = judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close)
    
    return {
        "politician_opening": pol_open,
        "scientist_opening": sci_open,
        "round_1_judge": round_1_judge,
        "politician_rebuttal": pol_rebut,
        "scientist_rebuttal": sci_rebut,
        "round_2_judge": round_2_judge,
        "politician_closing": pol_close,
        "scientist_closing": sci_close,
        "final_judge": final_judge
    }


def run_multi_agent_people_round_mcq(claim, evidence):
    """Run the debate and compute verdict token probabilities at each judge."""
    print("\n=== Running Multi-Agent People Debate with Round Judges (probability view) ===")

    print("Round 1: Opening statements...")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)

    print("Round 1 Judge evaluation (probabilities)...")
    round_1_judge = judge_round_1_mcq(claim, evidence, pol_open, sci_open)

    print("Round 2: Rebuttal statements...")
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)

    print("Round 2 Judge evaluation (probabilities)...")
    round_2_judge = judge_round_2_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)

    print("Round 3: Closing statements...")
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)

    print("Final Judge evaluation (probabilities)...")
    final_judge = judge_final_verdict_mcq(
        claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close
    )

    return {
        "politician_opening": pol_open,
        "scientist_opening": sci_open,
        "round_1_judge": round_1_judge,
        "politician_rebuttal": pol_rebut,
        "scientist_rebuttal": sci_rebut,
        "round_2_judge": round_2_judge,
        "politician_closing": pol_close,
        "scientist_closing": sci_close,
        "final_judge": final_judge
    }
