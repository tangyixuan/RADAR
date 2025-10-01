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
from typing import Dict

# Global model info - will be set by main.py
model_info = None


MCQ_VERDICT_MAP = {
    "A": "TRUE",
    "B": "FALSE",
    "C": "HALF-TRUE",
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


def _append_mcq_instructions(prompt: str) -> str:
    """Ensure the prompt includes MCQ instructions."""
    prompt = prompt.rstrip()
    instructions = (
        "\n\nChoose exactly one of the following options:\n"
        "A) TRUE\n"
        "B) FALSE\n"
        "C) HALF-TRUE\n"
        "Reply with the letter (A, B, or C) only."
    )
    return prompt + instructions


def _logprobs_to_probabilities(logprob_map: Dict[str, float]) -> Dict[str, float]:
    """Convert choice log-probabilities into a probability distribution."""
    max_logprob = max(logprob_map.values())
    exp_map = {choice: math.exp(logprob - max_logprob) for choice, logprob in logprob_map.items()}
    total = sum(exp_map.values())
    return {choice: (value / total if total > 0 else 0.0) for choice, value in exp_map.items()}


def _compute_mcq_choice_logprobs_local(tokenizer, model, system_prompt: str, user_prompt: str) -> Dict[str, float]:
    """Compute per-choice log-probabilities for local models."""
    text, _ = build_chat_prompt(tokenizer, system_prompt, user_prompt)
    base_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_length = base_inputs.input_ids.shape[1]

    choice_logprobs: Dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for choice in MCQ_VERDICT_MAP.keys():
            choice_text = text + choice
            choice_inputs = tokenizer([choice_text], return_tensors="pt").to(model.device)
            input_ids = choice_inputs.input_ids
            attention_mask = choice_inputs.get("attention_mask")

            labels = input_ids.clone()
            labels[:, :prompt_length] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            token_count = (labels != -100).sum().item()
            if token_count == 0:
                choice_logprobs[choice] = float("-inf")
            else:
                choice_logprobs[choice] = -outputs.loss.item() * token_count

    return choice_logprobs


def _compute_mcq_choice_logprobs_gpt(client, model_name: str, system_prompt: str, user_prompt: str) -> Dict[str, float]:
    """Compute per-choice log-probabilities for GPT-style models via echo logprobs."""
    base_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: "
    base_length = len(base_prompt)

    choice_logprobs: Dict[str, float] = {}
    for choice in MCQ_VERDICT_MAP.keys():
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

        logprob_sum = 0.0
        for token, token_logprob, offset in zip(
            logprob_data.tokens,
            logprob_data.token_logprobs,
            logprob_data.text_offset
        ):
            if offset is not None and offset >= base_length and token_logprob is not None:
                logprob_sum += token_logprob
        choice_logprobs[choice] = logprob_sum

    return choice_logprobs


def compute_mcq_choice_logprobs(system_prompt: str, user_prompt: str) -> Dict[str, float]:
    """Dispatch log-probability computation based on loaded model."""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")

    if len(model_info) != 2:
        raise ValueError("Invalid model_info format")

    first, second = model_info
    if isinstance(second, str):
        client, model_name = model_info
        return _compute_mcq_choice_logprobs_gpt(client, model_name, system_prompt, user_prompt)

    tokenizer, model = model_info
    return _compute_mcq_choice_logprobs_local(tokenizer, model, system_prompt, user_prompt)


def _build_mcq_judge_result(user_prompt: str) -> Dict[str, object]:
    """Run MCQ scoring for a judge prompt and package results."""
    system_prompt = get_system_prompt("judge")
    logprobs = compute_mcq_choice_logprobs(system_prompt, user_prompt)
    choice_probabilities = _logprobs_to_probabilities(logprobs)

    best_choice = max(choice_probabilities, key=choice_probabilities.get)
    verdict = MCQ_VERDICT_MAP[best_choice]

    verdict_probabilities = {
        MCQ_VERDICT_MAP[choice]: choice_probabilities[choice]
        for choice in MCQ_VERDICT_MAP
    }

    return {
        "response": best_choice,
        "verdict": verdict,
        "probability": verdict_probabilities[verdict],
        "choice_probabilities": choice_probabilities,
        "choice_logprobs": logprobs,
        "probabilities": verdict_probabilities
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
    prompt = _append_mcq_instructions(judge_prompt_1r(claim, evidence, pol_open, sci_open))
    return _build_mcq_judge_result(prompt)

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
    prompt = _append_mcq_instructions(
        judge_prompt_2r(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    )
    return _build_mcq_judge_result(prompt)

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
    prompt = _append_mcq_instructions(
        judge_prompt(
            claim, evidence,
            pol_open, sci_open,
            pol_rebut, sci_rebut,
            pol_close, sci_close
        )
    )
    return _build_mcq_judge_result(prompt)

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
    """Run the debate with MCQ-based judges to obtain per-verdict probabilities."""
    print("\n=== Running Multi-Agent People Debate with Round Judges (MCQ) ===")

    print("Round 1: Opening statements...")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)

    print("Round 1 Judge evaluation (MCQ)...")
    round_1_judge = judge_round_1_mcq(claim, evidence, pol_open, sci_open)

    print("Round 2: Rebuttal statements...")
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)

    print("Round 2 Judge evaluation (MCQ)...")
    round_2_judge = judge_round_2_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)

    print("Round 3: Closing statements...")
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)

    print("Final Judge evaluation (MCQ)...")
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
