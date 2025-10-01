from model.loader import load_model
from prompts.templates import system_prompt_fact_checker, user_prompt_single_agent
from agents.chat_template_utils import (
    build_chat_prompt,
    build_chat_prompts,
    extract_assistant_response,
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

def verify_claim(claim, evidence):
    """
    Verify the veracity of a given claim using retrieved evidence.
    Returns the model's classification and explanation.
    """
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    # Load prompt components
    system_prompt = system_prompt_fact_checker()
    user_prompt = user_prompt_single_agent(claim, evidence)

    # Generate response using the run_model function
    response = run_model(system_prompt, user_prompt)
    return response

def verify_claims_batch(claims, evidences):
    """
    Verify multiple claims in batch using retrieved evidence.
    Returns list of model's classifications and explanations.
    """
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    # Load prompt components
    system_prompt = system_prompt_fact_checker()
    system_prompts = [system_prompt] * len(claims)
    user_prompts = [user_prompt_single_agent(claim, evidence) for claim, evidence in zip(claims, evidences)]

    # Generate responses using batch processing
    responses = run_model_batch(system_prompts, user_prompts)
    return responses
