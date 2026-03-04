from model.loader import load_model
from prompts.templates import (
    get_system_prompt,
    user_prompt_opening_pro,
    user_prompt_rebuttal_pro,
    user_prompt_closing_pro,
    user_prompt_opening_con,
    user_prompt_rebuttal_con,
    user_prompt_closing_con,
    user_prompt_judge_full
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
        
        # Determine the model type by inspecting the second element
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
    
# === Pro Agent ===
def opening_pro(claim, evidence):
    prompt = user_prompt_opening_pro(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_pro(claim, evidence, con_opening_statement):
    prompt = user_prompt_rebuttal_pro(claim, evidence, con_opening_statement)
    return run_model(get_system_prompt("debater"), prompt)

def closing_pro(claim, evidence):
    prompt = user_prompt_closing_pro(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)


# === Con Agent ===
def opening_con(claim, evidence):
    prompt = user_prompt_opening_con(claim, evidence)
    return run_model(get_system_prompt("debater"), prompt)

def rebuttal_con(claim, evidence, pro_opening_statement):
    prompt = user_prompt_rebuttal_con(claim, evidence, pro_opening_statement)
    return run_model(get_system_prompt("debater"), prompt)

def closing_con(claim, evidence):
    prompt = user_prompt_closing_con(claim, evidence)
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
