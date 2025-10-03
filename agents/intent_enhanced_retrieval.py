from model.loader import load_model
from prompts.templates import (
    get_system_prompt,
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con
)
from agents.chat_template_utils import (
    build_chat_prompt,
    extract_assistant_response,
    inference_generate,
)

# Global variables for model and tokenizer
tokenizer = None
model = None

def set_model_info(model_info):
    """Set the global model and tokenizer"""
    global tokenizer, model
    if isinstance(model_info, tuple):
        tokenizer, model = model_info
    else:
        # For GPT models, model_info is (client, model_name)
        tokenizer, model = model_info, None

# Load model once for all agents (default to llama for backward compatibility)
if tokenizer is None and model is None:
    tokenizer, model = load_model(model_type="llama")

def infer_intent(claim):
    prompt = user_prompt_intent_inference(claim)
    return run_model(get_system_prompt("fact_checker"), prompt, max_tokens=150)

def reformulate_claim_pro(claim, intent):
    prompt = user_prompt_reformulate_pro(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=100)

def reformulate_claim_con(claim, intent):
    prompt = user_prompt_reformulate_con(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=100)

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

def intent_enhanced_reformulation(claim: str):

    # Step 1: Infer Intent
    intent = infer_intent(claim)
    # print(f"[Inferred Intent]: {intent}")

    # Step 2: Reformulate Pro Version
    reformulated_pro = reformulate_claim_pro(claim, intent)
    # print(f"[Pro Reformulated Claim]: {reformulated_pro}")

    # Step 3: Reformulate Con Version
    reformulated_con = reformulate_claim_con(claim, intent)
    # print(f"[Con Reformulated Claim]: {reformulated_con}")

    return {
        "intent": intent,
        "reformulated_pro": reformulated_pro,
        "reformulated_con": reformulated_con
    }
