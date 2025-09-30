from model.loader import load_model
from prompts.templates import system_prompt_fact_checker, user_prompt_single_agent

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
            if 'qwen' in tokenizer_class_name:
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
                # Llama model - 使用正确的 Llama 3 格式
                text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
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
                # Llama模型使用正确的格式
                if "<|start_header_id|>assistant<|end_header_id|>" in response:
                    return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                elif "<|assistant|>" in response:
                    return response.split("<|assistant|>")[-1].strip()
                else:
                    return response.strip()
    
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
                    text = f"<|begin_of_text|><|system|>\n{sys_prompt}\n<|user|>\n{usr_prompt}<|assistant|>\n"
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