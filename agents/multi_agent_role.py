from model.loader import load_model
from prompts.templates_role import (
    get_system_prompt,
    user_prompt_opening_pro,
    user_prompt_rebuttal_pro,
    user_prompt_closing_pro,
    user_prompt_opening_con,
    user_prompt_rebuttal_con,
    user_prompt_closing_con,
    user_prompt_judge_full,
    user_prompt_intent_inference,
    user_prompt_role_inference
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
                text = f"<|begin_of_text|><|system|>{system_prompt}<|user|>{user_prompt}<|assistant|>"
            
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
                if "assistant" in response:
                    return response.split("assistant")[-1].strip()
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
                    text = f"<|begin_of_text|><|system|>{sys_prompt}<|user|>{usr_prompt}<|assistant|>"
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
                    if "assistant" in response:
                        response = response.split("assistant")[-1].strip()
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

def run_multi_agent_role_batch(claims, evidences, batch_size=8):
    """Run multi-agent role-based debate in batch for Qwen model"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        
        batch_results = []
        for claim, evidence in zip(batch_claims, batch_evidences):
            # Step 1: Infer intent and roles
            intent, support_role, oppose_role = infer_intent_and_roles(claim)
            
            # Generate prompts
            pro_open_prompt = user_prompt_opening_pro(claim, evidence, support_role)
            con_open_prompt = user_prompt_opening_con(claim, evidence, oppose_role)
            
            # Batch process opening statements
            system_prompt = get_system_prompt("debater")
            opening_results = run_model_batch(
                [system_prompt] * 2, 
                [pro_open_prompt, con_open_prompt]
            )
            pro_open, con_open = opening_results
            
            # Generate rebuttal prompts
            pro_rebut_prompt = user_prompt_rebuttal_pro(claim, evidence, con_open, support_role)
            con_rebut_prompt = user_prompt_rebuttal_con(claim, evidence, pro_open, oppose_role)
            
            # Batch process rebuttals
            rebuttal_results = run_model_batch(
                [system_prompt] * 2,
                [pro_rebut_prompt, con_rebut_prompt]
            )
            pro_rebut, con_rebut = rebuttal_results
            
            # Generate closing prompts
            pro_close_prompt = user_prompt_closing_pro(claim, evidence, support_role)
            con_close_prompt = user_prompt_closing_con(claim, evidence, oppose_role)
            
            # Batch process closings
            closing_results = run_model_batch(
                [system_prompt] * 2,
                [pro_close_prompt, con_close_prompt]
            )
            pro_close, con_close = closing_results
            
            # Generate judge prompt
            judge_prompt = user_prompt_judge_full(
                claim, evidence,
                pro_open, con_open,
                pro_rebut, con_rebut,
                pro_close, con_close
            )
            
            # Process judge verdict
            judge_system = get_system_prompt("judge")
            final_result = run_model(judge_system, judge_prompt, max_tokens=400)
            
            batch_results.append({
                "intent": intent,
                "support_role": support_role,
                "oppose_role": oppose_role,
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            })
        
        results.extend(batch_results)
    
    return results

# === Step 1: Infer intent and roles ===
def infer_intent_and_roles(claim):
    intent_prompt = user_prompt_intent_inference(claim)
    intent = run_model(get_system_prompt("fact_checker"), intent_prompt)

    role_prompt = user_prompt_role_inference(intent)
    roles_output = run_model(get_system_prompt("fact_checker"), role_prompt)

    support_role = "Pro"
    oppose_role = "Con"
    for line in roles_output.splitlines():
        if line.startswith("SUPPORTING_ROLE:"):
            support_role = line.split(":", 1)[1].strip()
        elif line.startswith("OPPOSING_ROLE:"):
            oppose_role = line.split(":", 1)[1].strip()
    return intent, support_role, oppose_role

# === Pro Agent ===
def opening_pro(claim, evidence, role):
    prompt = user_prompt_opening_pro(claim, evidence, role)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_pro(claim, evidence, con_opening_statement, role):
    prompt = user_prompt_rebuttal_pro(claim, evidence, con_opening_statement, role)
    return run_model(get_system_prompt("debater"), prompt)

def closing_pro(claim, evidence, role):
    prompt = user_prompt_closing_pro(claim, evidence, role)
    return run_model(get_system_prompt("debater"), prompt)

# === Con Agent ===
def opening_con(claim, evidence, role):
    prompt = user_prompt_opening_con(claim, evidence, role)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_con(claim, evidence, pro_opening_statement, role):
    prompt = user_prompt_rebuttal_con(claim, evidence, pro_opening_statement, role)
    return run_model(get_system_prompt("debater"), prompt)

def closing_con(claim, evidence, role):
    prompt = user_prompt_closing_con(claim, evidence, role)
    return run_model(get_system_prompt("debater"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
    prompt = user_prompt_judge_full(
        claim, evidence,
        pro_open, con_open,
        pro_rebut, con_rebut,
        pro_close, con_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)