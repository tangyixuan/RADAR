from model.loader import load_model
from prompts.templates_party import (
    get_system_prompt,
    democrat_opening_prompt,
    democrat_rebuttal_prompt,
    democrat_closing_prompt,
    republican_opening_prompt,
    republican_rebuttal_prompt,
    republican_closing_prompt,
    judge_prompt
)

# Global model info - will be set by main.py
model_info = None

def set_model_info(info):
    """Set the global model info"""
    global model_info
    model_info = info

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
            
            # 通过tokenizer的类名或模型名称来区分Qwen和Llama
            tokenizer_class_name = tokenizer.__class__.__name__.lower()
            if 'qwen' in tokenizer_class_name or hasattr(tokenizer, 'apply_chat_template'):
                print("Qwen model")
                # Qwen model - 使用apply_chat_template
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                print("Llama model")
                # Llama model - 使用原始模板格式
                text = f"<|begin_of_text|><|system|>\n{system_prompt}\n<|user|>\n{user_prompt}<|assistant|>\n"
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 对于Qwen模型，需要根据实际输出格式来提取assistant回复
            if hasattr(tokenizer, 'apply_chat_template'):
                # Qwen模型使用简单的assistant标记
                if "assistant\n" in response:
                    return response.split("assistant\n")[-1].strip()
                elif "<|assistant|>" in response:
                    return response.split("<|assistant|>")[-1].strip()
                else:
                    return response.strip()
            else:
                # Llama模型使用原始格式
                return response.split("<|assistant|>")[-1].strip()
    
    else:
        raise ValueError("Invalid model_info format")

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
            full_prompts = []
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                # 通过tokenizer的类名来区分Qwen和Llama
                tokenizer_class_name = tokenizer.__class__.__name__.lower()
                if 'qwen' in tokenizer_class_name or hasattr(tokenizer, 'apply_chat_template'):
                    # Qwen model - 使用apply_chat_template
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Llama model - 使用原始模板格式
                    text = f"<|begin_of_text|><|system|>\\n{sys_prompt}\\n<|user|>\\n{usr_prompt}<|assistant|>\\n"
                full_prompts.append(text)
            
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
                response = tokenizer.decode(output, skip_special_tokens=True)
                # 对于Qwen模型，需要根据实际输出格式来提取assistant回复
                if hasattr(tokenizer, 'apply_chat_template'):
                    # Qwen模型使用简单的assistant标记
                    if "assistant\n" in response:
                        response = response.split("assistant\n")[-1].strip()
                    elif "<|assistant|>" in response:
                        response = response.split("<|assistant|>")[-1].strip()
                    else:
                        response = response.strip()
                else:
                    # Llama模型使用原始格式
                    if "<|assistant|>" in response:
                        response = response.split("<|assistant|>")[-1].strip()
                responses.append(response)
            
            return responses
    
    else:
        raise ValueError("Invalid model_info format")

def run_multi_agent_party_batch(claims, evidences, batch_size=8):
    """Run multi-agent party debate in batch for Qwen model"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        
        batch_results = []
        for claim, evidence in zip(batch_claims, batch_evidences):
            # Generate prompts
            dem_open_prompt = democrat_opening_prompt(claim, evidence)
            rep_open_prompt = republican_opening_prompt(claim, evidence)
            
            # Batch process opening statements
            dem_system = get_system_prompt("democrat")
            rep_system = get_system_prompt("republican")
            opening_results = run_model_batch(
                [dem_system, rep_system], 
                [dem_open_prompt, rep_open_prompt]
            )
            dem_open, rep_open = opening_results
            
            # Generate rebuttal prompts
            dem_rebut_prompt = democrat_rebuttal_prompt(claim, evidence, rep_open)
            rep_rebut_prompt = republican_rebuttal_prompt(claim, evidence, dem_open)
            
            # Batch process rebuttals
            rebuttal_results = run_model_batch(
                [dem_system, rep_system],
                [dem_rebut_prompt, rep_rebut_prompt]
            )
            dem_rebut, rep_rebut = rebuttal_results
            
            # Generate closing prompts
            dem_close_prompt = democrat_closing_prompt(claim, evidence)
            rep_close_prompt = republican_closing_prompt(claim, evidence)
            
            # Batch process closings
            closing_results = run_model_batch(
                [dem_system, rep_system],
                [dem_close_prompt, rep_close_prompt]
            )
            dem_close, rep_close = closing_results
            
            # Generate judge prompt
            judge_prompt_text = judge_prompt(
                claim, evidence,
                dem_open, rep_open,
                dem_rebut, rep_rebut,
                dem_close, rep_close
            )
            
            # Process judge verdict
            judge_system = get_system_prompt("judge")
            final_result = run_model(judge_system, judge_prompt_text, max_tokens=400)
            
            batch_results.append({
                "democrat_opening": dem_open,
                "republican_opening": rep_open,
                "democrat_rebuttal": dem_rebut,
                "republican_rebuttal": rep_rebut,
                "democrat_closing": dem_close,
                "republican_closing": rep_close,
                "final_verdict": final_result
            })
        
        results.extend(batch_results)
    
    return results

# === Democrat Agent ===
def opening_democrat(claim, evidence):
    prompt = democrat_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("democrat"), prompt)

def rebuttal_democrat(claim, evidence, opponent_argument):
    prompt = democrat_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("democrat"), prompt)

def closing_democrat(claim, evidence):
    prompt = democrat_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("democrat"), prompt)

# === Republican Agent ===
def opening_republican(claim, evidence):
    prompt = republican_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("republican"), prompt)

def rebuttal_republican(claim, evidence, opponent_argument):
    prompt = republican_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("republican"), prompt)

def closing_republican(claim, evidence):
    prompt = republican_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("republican"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, dem_open, rep_open, dem_rebut, rep_rebut, dem_close, rep_close):
    prompt = judge_prompt(
        claim, evidence,
        dem_open, rep_open,
        dem_rebut, rep_rebut,
        dem_close, rep_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)