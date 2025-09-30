
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
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con
)

# Load model once for all agents
tokenizer, model = load_model()

# Global variable to store model info
model_info = None

def set_model_info(info):
    """Set the model info globally"""
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

# === Intent Inference and Reformulation ===
def infer_intent(claim):
    """Infer the intended message of a claim"""
    prompt = user_prompt_intent_inference(claim)
    return run_model(get_system_prompt("fact_checker"), prompt, max_tokens=100)

def reformulate_claim_pro(claim, intent):
    """Reformulate claim from pro perspective"""
    prompt = user_prompt_reformulate_pro(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=150)

def reformulate_claim_con(claim, intent):
    """Reformulate claim from con perspective"""
    prompt = user_prompt_reformulate_con(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=150)

def intent_enhanced_reformulation(claim: str):
    """Perform intent-enhanced claim reformulation"""
    # Step 1: Infer Intent
    intent = infer_intent(claim)
    
    # Step 2: Reformulate Pro Version
    reformulated_pro = reformulate_claim_pro(claim, intent)
    
    # Step 3: Reformulate Con Version
    reformulated_con = reformulate_claim_con(claim, intent)
    
    return {
        "intent": intent,
        "reformulated_pro": reformulated_pro,
        "reformulated_con": reformulated_con
    }

# === Politician Agent ===
def opening_politician(claim, evidence):
    """Politician opening with reformulated pro claim"""
    prompt = politician_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, opponent_argument):
    """Politician rebuttal with reformulated pro claim"""
    prompt = politician_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence):
    """Politician closing with reformulated pro claim"""
    prompt = politician_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence):
    """Scientist opening with reformulated con claim"""
    prompt = scientist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, opponent_argument):
    """Scientist rebuttal with reformulated con claim"""
    prompt = scientist_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence):
    """Scientist closing with reformulated con claim"""
    prompt = scientist_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

# === Judge Agent ===
def judge_final_verdict(original_claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    """Judge verdict considering intent and reformulated claims"""
    # Use the imported judge_prompt function directly
    prompt = judge_prompt(original_claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close)
    
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

# === Multi-Agent Functions ===
def run_multi_agent_people(claim, evidence):
    """Run multi-agent people debate with intent-enhanced reformulation"""
    # Step 1: Perform intent-enhanced reformulation
    reformulation_result = intent_enhanced_reformulation(claim)
    intent = reformulation_result["intent"]
    reformulated_pro = reformulation_result["reformulated_pro"]
    reformulated_con = reformulation_result["reformulated_con"]
    
    # Step 2: Politician uses pro reformulation, Scientist uses con reformulation
    pol_open = opening_politician(reformulated_pro, evidence)
    sci_open = opening_scientist(reformulated_con, evidence)
    
    pol_rebut = rebuttal_politician(reformulated_pro, evidence, sci_open)
    sci_rebut = rebuttal_scientist(reformulated_con, evidence, pol_open)
    
    pol_close = closing_politician(reformulated_pro, evidence)
    sci_close = closing_scientist(reformulated_con, evidence)
    
    # Step 3: Judge evaluates with original claim but reformulated arguments
    final_verdict = judge_final_verdict(
        claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close
    )
    
    return {
        "original_claim": claim,
        "intent": intent,
        "reformulated_pro": reformulated_pro,
        "reformulated_con": reformulated_con,
        "politician_opening": pol_open,
        "scientist_opening": sci_open,
        "politician_rebuttal": pol_rebut,
        "scientist_rebuttal": sci_rebut,
        "politician_closing": pol_close,
        "scientist_closing": sci_close,
        "final_verdict": final_verdict
    }

def run_multi_agent_people_batch(claims, evidences, batch_size=8):
    """Run multi-agent people debate with intent-enhanced reformulation in batch"""
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        
        batch_results = []
        for claim, evidence in zip(batch_claims, batch_evidences):
            result = run_multi_agent_people(claim, evidence)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results