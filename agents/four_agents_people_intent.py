from model.loader import load_model
from prompts.templates_four_people import (
    get_system_prompt,
    user_prompt_opening_politician,
    user_prompt_rebuttal_politician,
    user_prompt_closing_politician,
    user_prompt_opening_scientist,
    user_prompt_rebuttal_scientist,
    user_prompt_closing_scientist,
    user_prompt_opening_journalist,
    user_prompt_rebuttal_journalist,
    user_prompt_closing_journalist,
    user_prompt_opening_domain_scientist,
    user_prompt_rebuttal_domain_scientist,
    user_prompt_closing_domain_scientist,
    user_prompt_judge_4_agents,
    user_prompt_domain_inference,
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

# === Domain Specialist Inference ===
def infer_domain_specialist(claim):
    prompt = user_prompt_domain_inference(claim)
    domain_output = run_model(get_system_prompt("fact_checker"), prompt, max_tokens=100)
    
    # Parse domain specialist from output
    domain_specialist = "Domain Expert"
    for line in domain_output.splitlines():
        if line.startswith("DOMAIN:"):
            domain_specialist = line.split(":", 1)[1].strip()
            break
    
    return domain_specialist

# === Individual Agent Functions ===
# === Politician Agent ===
def opening_politician(claim, evidence):
    prompt = user_prompt_opening_politician(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, scientist_argument, journalist_argument, domain_scientist_argument):
    prompt = user_prompt_rebuttal_politician(claim, evidence, scientist_argument, journalist_argument, domain_scientist_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence):
    prompt = user_prompt_closing_politician(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence):
    prompt = user_prompt_opening_scientist(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, politician_argument, journalist_argument, domain_scientist_argument):
    prompt = user_prompt_rebuttal_scientist(claim, evidence, politician_argument, journalist_argument, domain_scientist_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence):
    prompt = user_prompt_closing_scientist(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

# === Journalist Agent ===
def opening_journalist(claim, evidence):
    prompt = user_prompt_opening_journalist(claim, evidence)
    return run_model(get_system_prompt("journalist"), prompt)

def rebuttal_journalist(claim, evidence, politician_argument, scientist_argument, domain_scientist_argument):
    prompt = user_prompt_rebuttal_journalist(claim, evidence, politician_argument, scientist_argument, domain_scientist_argument)
    return run_model(get_system_prompt("journalist"), prompt)

def closing_journalist(claim, evidence):
    prompt = user_prompt_closing_journalist(claim, evidence)
    return run_model(get_system_prompt("journalist"), prompt)

# === Domain Scientist Agent ===
def opening_domain_scientist(claim, evidence, domain_specialist: str = None):
    prompt = user_prompt_opening_domain_scientist(claim, evidence, domain_specialist)
    return run_model(get_system_prompt("domain_scientist", domain_specialist), prompt)

def rebuttal_domain_scientist(claim, evidence, politician_argument, scientist_argument, journalist_argument, domain_specialist: str = None):
    prompt = user_prompt_rebuttal_domain_scientist(claim, evidence, politician_argument, scientist_argument, journalist_argument, domain_specialist)
    return run_model(get_system_prompt("domain_scientist", domain_specialist), prompt)

def closing_domain_scientist(claim, evidence, domain_specialist: str = None):
    prompt = user_prompt_closing_domain_scientist(claim, evidence, domain_specialist)
    return run_model(get_system_prompt("domain_scientist", domain_specialist), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, politician_open, scientist_open, journalist_open, domain_scientist_open, 
                       politician_rebut, scientist_rebut, journalist_rebut, domain_scientist_rebut,
                       politician_close, scientist_close, journalist_close, domain_scientist_close):
    prompt = user_prompt_judge_4_agents(
        claim, evidence, politician_open, scientist_open, journalist_open, domain_scientist_open,
        politician_rebut, scientist_rebut, journalist_rebut, domain_scientist_rebut,
        politician_close, scientist_close, journalist_close, domain_scientist_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)

def run_four_agents_people_intent(claim, evidence):
    """Run 4-agent people debate with intent-enhanced reformulation"""
    # Step 1: Perform intent-enhanced reformulation
    reformulation_result = intent_enhanced_reformulation(claim)
    intent = reformulation_result["intent"]
    reformulated_pro = reformulation_result["reformulated_pro"]
    reformulated_con = reformulation_result["reformulated_con"]
    
    # Step 2: Infer domain specialist for this claim
    domain_specialist = infer_domain_specialist(claim)
    
    # Step 3: Opening statements (politician uses reformulated pro, scientist uses reformulated con)
    pol_open = opening_politician(reformulated_pro, evidence)
    sci_open = opening_scientist(reformulated_con, evidence)
    jour_open = opening_journalist(claim, evidence)  # Journalist uses original claim
    dom_open = opening_domain_scientist(claim, evidence, domain_specialist)  # Domain scientist uses original claim
    
    # Step 4: Rebuttals
    pol_rebut = rebuttal_politician(reformulated_pro, evidence, sci_open, jour_open, dom_open)
    sci_rebut = rebuttal_scientist(reformulated_con, evidence, pol_open, jour_open, dom_open)
    jour_rebut = rebuttal_journalist(claim, evidence, pol_open, sci_open, dom_open)
    dom_rebut = rebuttal_domain_scientist(claim, evidence, pol_open, sci_open, jour_open, domain_specialist)
    
    # Step 5: Closings
    pol_close = closing_politician(reformulated_pro, evidence)
    sci_close = closing_scientist(reformulated_con, evidence)
    jour_close = closing_journalist(claim, evidence)
    dom_close = closing_domain_scientist(claim, evidence, domain_specialist)
    
    # Step 6: Judge verdict
    final_result = judge_final_verdict(
        claim, evidence, 
        pol_open, sci_open, jour_open, dom_open,
        pol_rebut, sci_rebut, jour_rebut, dom_rebut,
        pol_close, sci_close, jour_close, dom_close
    )
    
    return {
        "original_claim": claim,
        "intent": intent,
        "reformulated_pro": reformulated_pro,
        "reformulated_con": reformulated_con,
        "domain_specialist": domain_specialist,
        "politician_opening": pol_open,
        "scientist_opening": sci_open,
        "journalist_opening": jour_open,
        "domain_scientist_opening": dom_open,
        "politician_rebuttal": pol_rebut,
        "scientist_rebuttal": sci_rebut,
        "journalist_rebuttal": jour_rebut,
        "domain_scientist_rebuttal": dom_rebut,
        "politician_closing": pol_close,
        "scientist_closing": sci_close,
        "journalist_closing": jour_close,
        "domain_scientist_closing": dom_close,
        "final_verdict": final_result
    } 
