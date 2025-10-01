
from model.loader import load_model
from prompts.templates_people import (
    get_system_prompt,
    politician_opening_prompt,
    politician_rebuttal_prompt,
    politician_cross_examination_prompt,
    politician_closing_prompt,
    scientist_opening_prompt,
    scientist_rebuttal_prompt,
    scientist_cross_examination_prompt,
    scientist_closing_prompt,
    judge_prompt_4r
)
from agents.chat_template_utils import (
    build_chat_prompt,
    build_chat_prompts,
    extract_assistant_response,
)

# Global model info
model_info = None

def set_model_info(info):
    """Set the global model info"""
    global model_info
    model_info = info

def run_model_batch(system_prompts: list, user_prompts: list, max_tokens: int = 300):
    """Run model inference in batch for Qwen model"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    if len(model_info) == 2:
        first, second = model_info
        if hasattr(first, 'chat') and hasattr(first.chat, 'completions'):
            # GPT model - process individually
            results = []
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                result = run_model(sys_prompt, usr_prompt, max_tokens)
                results.append(result)
            return results
        else:
            # Local model - batch processing
            tokenizer, model = model_info
            full_prompts, used_chat_templates = build_chat_prompts(tokenizer, system_prompts, user_prompts)
            
            # Tokenize all prompts
            inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            # Generate in batch
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Decode all outputs
            responses = []
            for i, output in enumerate(outputs):
                raw_response = tokenizer.decode(output, skip_special_tokens=True)
                responses.append(extract_assistant_response(raw_response, used_chat_templates[i]))
            
            return responses
    
    else:
        raise ValueError("Invalid model_info format")

def run_multi_agent_people_batch(claims, evidences, batch_size=8):
    """Run multi-agent people debate in batch for Qwen model"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        
        batch_results = []
        for claim, evidence in zip(batch_claims, batch_evidences):
            # Generate prompts
            pol_open_prompt = politician_opening_prompt(claim, evidence)
            sci_open_prompt = scientist_opening_prompt(claim, evidence)
            
            # Batch process opening statements
            pol_system = get_system_prompt("politician")
            sci_system = get_system_prompt("scientist")
            opening_results = run_model_batch(
                [pol_system, sci_system], 
                [pol_open_prompt, sci_open_prompt]
            )
            pol_open, sci_open = opening_results
            
            # Generate rebuttal prompts
            pol_rebut_prompt = politician_rebuttal_prompt(claim, evidence, sci_open)
            sci_rebut_prompt = scientist_rebuttal_prompt(claim, evidence, pol_open)
            
            # Batch process rebuttals
            rebuttal_results = run_model_batch(
                [pol_system, sci_system],
                [pol_rebut_prompt, sci_rebut_prompt]
            )
            pol_rebut, sci_rebut = rebuttal_results
            
            # Generate cross-examination prompts
            pol_cross_prompt = politician_cross_examination_prompt(claim, evidence, sci_rebut)
            sci_cross_prompt = scientist_cross_examination_prompt(claim, evidence, pol_rebut)
            
            # Batch process cross-examinations
            cross_results = run_model_batch(
                [pol_system, sci_system],
                [pol_cross_prompt, sci_cross_prompt]
            )
            pol_cross, sci_cross = cross_results
            
            # Generate closing prompts
            pol_close_prompt = politician_closing_prompt(claim, evidence)
            sci_close_prompt = scientist_closing_prompt(claim, evidence)
            
            # Batch process closings
            closing_results = run_model_batch(
                [pol_system, sci_system],
                [pol_close_prompt, sci_close_prompt]
            )
            pol_close, sci_close = closing_results
            
            # Generate judge prompt
            judge_prompt_text = judge_prompt_4r(
                claim, evidence,
                pol_open, sci_open,
                pol_rebut, sci_rebut,
                pol_cross, sci_cross,
                pol_close, sci_close
            )
            
            # Process judge verdict
            judge_system = get_system_prompt("judge")
            final_result = run_model(judge_system, judge_prompt_text, max_tokens=400)
            
            batch_results.append({
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "politician_cross_examination": pol_cross,
                "scientist_cross_examination": sci_cross,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "final_verdict": final_result
            })
        
        results.extend(batch_results)
    
    return results

# === Politician Agent ===
def opening_politician(claim, evidence):
    prompt = politician_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, opponent_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("politician"), prompt)

def cross_examination_politician(claim, evidence, opponent_argument):
    prompt = politician_cross_examination_prompt(claim, evidence, opponent_argument)
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

def cross_examination_scientist(claim, evidence, opponent_argument):
    prompt = scientist_cross_examination_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence):
    prompt = scientist_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_cross, sci_cross, pol_close, sci_close):
    prompt = judge_prompt_4r(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_cross, sci_cross,
        pol_close, sci_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

# === Shared model runner ===
def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300):
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
