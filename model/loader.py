from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def load_model(model_path=None, model_type="llama", api_key=None, gpt_model_name="gpt-4o-mini"):
    """
    Load model based on type
    
    Args:
        model_path: Path to local model (for llama or qwen)
        model_type: "llama", "qwen", or "gpt"
        api_key: OpenAI API key (required for gpt model)
        gpt_model_name: GPT model name (default: gpt-4o-mini)
    """
    if model_type == "llama":
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = "/home/yirui/mad_formal/model/llama3-8b-instruct/"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return tokenizer, model
    
    elif model_type == "qwen":
        if model_path is None:
            model_path = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return tokenizer, model
    
    elif model_type == "gpt":
        if api_key is None:
            api_key = os.getenv("single_full")
            if api_key is None:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file or pass it as api_key parameter")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # For GPT, we return a special tuple that includes the client
            return client, gpt_model_name
        except ImportError:
            raise ImportError("OpenAI module not found. Please install it with: pip install openai")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")