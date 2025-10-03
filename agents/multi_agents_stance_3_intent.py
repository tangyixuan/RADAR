from model.loader import load_model
from prompts.templates_stance_3 import (
    get_system_prompt,
    user_prompt_opening_pro,
    user_prompt_rebuttal_pro,
    user_prompt_closing_pro,
    user_prompt_opening_con,
    user_prompt_rebuttal_con,
    user_prompt_closing_con,
    user_prompt_opening_flexible,
    user_prompt_rebuttal_flexible,
    user_prompt_closing_flexible,
    judge_prompt_three_agents,
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con
)
from agents.chat_template_utils import (
    build_chat_prompt,
    extract_assistant_response,
    inference_generate,
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

            text, used_chat_template = build_chat_prompt(tokenizer, system_prompt, user_prompt)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "use_cache": True,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
            }
            outputs = inference_generate(
                model,
                inputs,
                **generate_kwargs,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return extract_assistant_response(response, used_chat_template)
    
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

# === Pro Agent ===
def opening_pro(claim, evidence):
    prompt = user_prompt_opening_pro(claim, evidence)
    return run_model(get_system_prompt("pro"), prompt)

def rebuttal_pro(claim, evidence, con_argument):
    prompt = user_prompt_rebuttal_pro(claim, evidence, con_argument)
    return run_model(get_system_prompt("pro"), prompt)

def closing_pro(claim, evidence):
    prompt = user_prompt_closing_pro(claim, evidence)
    return run_model(get_system_prompt("pro"), prompt)

# === Con Agent ===
def opening_con(claim, evidence):
    prompt = user_prompt_opening_con(claim, evidence)
    return run_model(get_system_prompt("con"), prompt)

def rebuttal_con(claim, evidence, pro_argument):
    prompt = user_prompt_rebuttal_con(claim, evidence, pro_argument)
    return run_model(get_system_prompt("con"), prompt)

def closing_con(claim, evidence):
    prompt = user_prompt_closing_con(claim, evidence)
    return run_model(get_system_prompt("con"), prompt)

# === Flexible Agent ===
def opening_flexible(claim, evidence, pro_argument, con_argument):
    prompt = user_prompt_opening_flexible(claim, evidence, pro_argument, con_argument)
    return run_model(get_system_prompt("flexible"), prompt)

def rebuttal_flexible(claim, evidence, pro_argument, con_argument):
    prompt = user_prompt_rebuttal_flexible(claim, evidence, pro_argument, con_argument)
    return run_model(get_system_prompt("flexible"), prompt)

def closing_flexible(claim, evidence, pro_argument, con_argument):
    prompt = user_prompt_closing_flexible(claim, evidence, pro_argument, con_argument)
    return run_model(get_system_prompt("flexible"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, flexible_open, pro_open, con_open, flexible_rebut, pro_rebut, con_rebut, flexible_close, pro_close, con_close):
    prompt = judge_prompt_three_agents(
        claim, evidence,
        flexible_open, pro_open, con_open,
        flexible_rebut, pro_rebut, con_rebut,
        flexible_close, pro_close, con_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

def run_multi_agent_stance_3_intent(claim, evidence):
    """Run multi-agent stance 3 debate with intent-enhanced reformulation"""
    # Step 1: Perform intent-enhanced reformulation
    reformulation_result = intent_enhanced_reformulation(claim)
    intent = reformulation_result["intent"]
    reformulated_pro = reformulation_result["reformulated_pro"]
    reformulated_con = reformulation_result["reformulated_con"]
    
    # Step 2: Generate opening statements for pro and con first (using reformulated claims)
    pro_open = opening_pro(reformulated_pro, evidence)
    con_open = opening_con(reformulated_con, evidence)
    
    # Step 3: Generate flexible opening statement (needs pro and con arguments)
    flex_open = opening_flexible(claim, evidence, pro_open, con_open)
    
    # Step 4: Generate rebuttals
    pro_rebut = rebuttal_pro(reformulated_pro, evidence, con_open)
    con_rebut = rebuttal_con(reformulated_con, evidence, pro_open)
    
    # Step 5: Generate flexible rebuttal (needs pro and con arguments)
    flex_rebut = rebuttal_flexible(claim, evidence, pro_rebut, con_rebut)
    
    # Step 6: Generate closings
    pro_close = closing_pro(reformulated_pro, evidence)
    con_close = closing_con(reformulated_con, evidence)
    
    # Step 7: Generate flexible closing (needs pro and con arguments)
    flex_close = closing_flexible(claim, evidence, pro_close, con_close)
    
    # Step 8: Judge verdict
    final_result = judge_final_verdict(
        claim, evidence,
        flex_open, pro_open, con_open,
        flex_rebut, pro_rebut, con_rebut,
        flex_close, pro_close, con_close
    )
    
    return {
        "original_claim": claim,
        "intent": intent,
        "reformulated_pro": reformulated_pro,
        "reformulated_con": reformulated_con,
        "flexible_opening": flex_open,
        "pro_opening": pro_open,
        "con_opening": con_open,
        "flexible_rebuttal": flex_rebut,
        "pro_rebuttal": pro_rebut,
        "con_rebuttal": con_rebut,
        "flexible_closing": flex_close,
        "pro_closing": pro_close,
        "con_closing": con_close,
        "final_verdict": final_result
    } 
