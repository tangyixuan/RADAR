
from model.loader import load_model
from prompts.templates_people_3 import (
    get_system_prompt,
    user_prompt_intent_inference,
    user_prompt_reformulate_pro,
    user_prompt_reformulate_con,
    politician_opening_prompt,
    politician_rebuttal_prompt,
    politician_closing_prompt,
    scientist_opening_prompt,
    scientist_rebuttal_prompt,
    scientist_closing_prompt,
    journalist_opening_prompt,
    journalist_rebuttal_prompt,
    journalist_closing_prompt,
    judge_prompt_three_agents
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

# === Claim Reformulation ===
def reformulate_claim_pro(claim, intent):
    """Reformulate claim from pro perspective"""
    prompt = user_prompt_reformulate_pro(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=150)

def reformulate_claim_con(claim, intent):
    """Reformulate claim from con perspective"""
    prompt = user_prompt_reformulate_con(claim, intent)
    return run_model(get_system_prompt("debater"), prompt, max_tokens=150)

# === Journalist Agent ===
def opening_journalist(claim, evidence):
    prompt = journalist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("journalist"), prompt)

def rebuttal_journalist(claim, evidence, politician_argument, scientist_argument):
    prompt = journalist_rebuttal_prompt(claim, evidence, politician_argument, scientist_argument)
    return run_model(get_system_prompt("journalist"), prompt)

def closing_journalist(claim, evidence, politician_rebuttal, scientist_rebuttal):
    prompt = journalist_closing_prompt(claim, evidence, politician_rebuttal, scientist_rebuttal)
    return run_model(get_system_prompt("journalist"), prompt)

# === Politician Agent ===
def opening_politician(claim, evidence, journalist_argument):
    prompt = politician_opening_prompt(claim, evidence, journalist_argument)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, scientist_argument, journalist_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, scientist_argument, journalist_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence, journalist_rebuttal):
    prompt = politician_closing_prompt(claim, evidence, journalist_rebuttal)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence, journalist_argument):
    prompt = scientist_opening_prompt(claim, evidence, journalist_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, politician_argument, journalist_argument):
    prompt = scientist_rebuttal_prompt(claim, evidence, politician_argument, journalist_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence, journalist_rebuttal):
    prompt = scientist_closing_prompt(claim, evidence, journalist_rebuttal)
    return run_model(get_system_prompt("scientist"), prompt)

# === Judge Agent ===
def judge_final_verdict(claim, evidence, jour_open, pol_open, sci_open, jour_rebut, pol_rebut, sci_rebut, jour_close, pol_close, sci_close):
    prompt = judge_prompt_three_agents(
        claim, evidence,
        jour_open, pol_open, sci_open,
        jour_rebut, pol_rebut, sci_rebut,
        jour_close, pol_close, sci_close
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)