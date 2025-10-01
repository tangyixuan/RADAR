from model.loader import load_model
from prompts.templates_four import (
    get_system_prompt,
    user_prompt_opening_pro1,
    user_prompt_opening_pro2,
    user_prompt_opening_con1,
    user_prompt_opening_con2,
    user_prompt_rebuttal_pro1,
    user_prompt_rebuttal_pro2,
    user_prompt_rebuttal_con1,
    user_prompt_rebuttal_con2,
    user_prompt_closing_pro1,
    user_prompt_closing_pro2,
    user_prompt_closing_con1,
    user_prompt_closing_con2,
    user_prompt_judge_4_agents,
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con
)
from agents.chat_template_utils import build_chat_prompt, extract_assistant_response

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

# === Individual Agent Functions ===
# === Pro-1 Agent (Factual Expert) ===
def opening_pro1(claim, evidence):
    prompt = user_prompt_opening_pro1(claim, evidence)
    return run_model(get_system_prompt("pro1"), prompt)

def rebuttal_pro1(claim, evidence, con1_argument, con2_argument):
    prompt = user_prompt_rebuttal_pro1(claim, evidence, con1_argument, con2_argument)
    return run_model(get_system_prompt("pro1"), prompt)

def closing_pro1(claim, evidence):
    prompt = user_prompt_closing_pro1(claim, evidence)
    return run_model(get_system_prompt("pro1"), prompt)

# === Pro-2 Agent (Reasoning Expert) ===
def opening_pro2(claim, evidence):
    prompt = user_prompt_opening_pro2(claim, evidence)
    return run_model(get_system_prompt("pro2"), prompt)

def rebuttal_pro2(claim, evidence, con1_argument, con2_argument):
    prompt = user_prompt_rebuttal_pro2(claim, evidence, con1_argument, con2_argument)
    return run_model(get_system_prompt("pro2"), prompt)

def closing_pro2(claim, evidence):
    prompt = user_prompt_closing_pro2(claim, evidence)
    return run_model(get_system_prompt("pro2"), prompt)

# === Con-1 Agent (Source Critic) ===
def opening_con1(claim, evidence):
    prompt = user_prompt_opening_con1(claim, evidence)
    return run_model(get_system_prompt("con1"), prompt)

def rebuttal_con1(claim, evidence, pro1_argument, pro2_argument):
    prompt = user_prompt_rebuttal_con1(claim, evidence, pro1_argument, pro2_argument)
    return run_model(get_system_prompt("con1"), prompt)

def closing_con1(claim, evidence):
    prompt = user_prompt_closing_con1(claim, evidence)
    return run_model(get_system_prompt("con1"), prompt)

# === Con-2 Agent (Reasoning Critic) ===
def opening_con2(claim, evidence):
    prompt = user_prompt_opening_con2(claim, evidence)
    return run_model(get_system_prompt("con2"), prompt)

def rebuttal_con2(claim, evidence, pro1_argument, pro2_argument):
    prompt = user_prompt_rebuttal_con2(claim, evidence, pro1_argument, pro2_argument)
    return run_model(get_system_prompt("con2"), prompt)

def closing_con2(claim, evidence):
    prompt = user_prompt_closing_con2(claim, evidence)
    return run_model(get_system_prompt("con2"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, pro1_open, pro2_open, con1_open, con2_open, 
                       pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
                       pro1_close, pro2_close, con1_close, con2_close):
    prompt = user_prompt_judge_4_agents(
        claim, evidence,
        pro1_open, pro2_open, con1_open, con2_open,
        pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
        pro1_close, pro2_close, con1_close, con2_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

def run_four_agents_intent(claim, evidence):
    """Run 4-agent debate with intent-enhanced reformulation"""
    # Step 1: Perform intent-enhanced reformulation
    reformulation_result = intent_enhanced_reformulation(claim)
    intent = reformulation_result["intent"]
    reformulated_pro = reformulation_result["reformulated_pro"]
    reformulated_con = reformulation_result["reformulated_con"]
    
    # Step 2: Opening statements (pro agents use reformulated pro claim, con agents use reformulated con claim)
    pro1_open = opening_pro1(reformulated_pro, evidence)
    pro2_open = opening_pro2(reformulated_pro, evidence)
    con1_open = opening_con1(reformulated_con, evidence)
    con2_open = opening_con2(reformulated_con, evidence)
    
    # Step 3: Rebuttals
    pro1_rebut = rebuttal_pro1(reformulated_pro, evidence, con1_open, con2_open)
    pro2_rebut = rebuttal_pro2(reformulated_pro, evidence, con1_open, con2_open)
    con1_rebut = rebuttal_con1(reformulated_con, evidence, pro1_open, pro2_open)
    con2_rebut = rebuttal_con2(reformulated_con, evidence, pro1_open, pro2_open)
    
    # Step 4: Closings
    pro1_close = closing_pro1(reformulated_pro, evidence)
    pro2_close = closing_pro2(reformulated_pro, evidence)
    con1_close = closing_con1(reformulated_con, evidence)
    con2_close = closing_con2(reformulated_con, evidence)
    
    # Step 5: Judge verdict
    final_result = judge_final_verdict(
        claim, evidence,
        pro1_open, pro2_open, con1_open, con2_open,
        pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
        pro1_close, pro2_close, con1_close, con2_close
    )
    
    return {
        "original_claim": claim,
        "intent": intent,
        "reformulated_pro": reformulated_pro,
        "reformulated_con": reformulated_con,
        "pro1_opening": pro1_open,
        "pro2_opening": pro2_open,
        "con1_opening": con1_open,
        "con2_opening": con2_open,
        "pro1_rebuttal": pro1_rebut,
        "pro2_rebuttal": pro2_rebut,
        "con1_rebuttal": con1_rebut,
        "con2_rebuttal": con2_rebut,
        "pro1_closing": pro1_close,
        "pro2_closing": pro2_close,
        "con1_closing": con1_close,
        "con2_closing": con2_close,
        "final_verdict": final_result
    } 
