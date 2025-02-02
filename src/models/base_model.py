# Pretrained LLM wrapper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseLLM:
    """
    Wrapper around a pretrained large language model (LLM) with inference utilities.
    """
    def __init__(self, model_name="meta-llama/Llama-2-7b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate_response(self, input_text, max_length=256, temperature=0.7, top_p=0.9):
        """
        Generate text from the model given an input prompt.
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def batch_generate(self, input_texts, max_length=256):
        """
        Batch generate responses for multiple input prompts.
        """
        input_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
        output_ids = self.model.generate(input_ids, max_length=max_length)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

    def evaluate_response(self, input_text, expected_output):
        """
        Compute loss between model-generated response and expected output.
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(expected_output, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            loss = outputs.loss
        return loss.item()

if __name__ == "__main__":
    model = BaseLLM()
    print(model.generate_response("What is the capital of France?"))
