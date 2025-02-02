# Main training script
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.assc_model import ASSCModel
from src.datasets.data_loader import get_training_data

class ASSCTrainer:
    def __init__(self, model_name="meta-llama/Llama-2-7b", batch_size=4, learning_rate=2e-5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ASSCModel(model_name).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.train_loader = DataLoader(get_training_data(), batch_size=batch_size, shuffle=True)

    def compute_reward(self, response, ground_truth):
        """ Multi-objective reward: truthfulness + consistency + ethical safety """
        truthfulness_score = self.model.estimate_response_confidence(response)
        consistency_score = 1 - abs(truthfulness_score - self.model.estimate_response_confidence(ground_truth))
        return 0.5 * truthfulness_score + 0.5 * consistency_score  # Weighted sum

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            input_texts, expected_responses = batch
            inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            targets = self.tokenizer(expected_responses, return_tensors="pt", padding=True, truncation=True).to(self.device)

            outputs = self.model.model(**inputs, labels=targets["input_ids"])
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss/len(self.train_loader):.4f}")

    def train(self, epochs=3):
        for epoch in range(epochs):
            self.train_epoch(epoch)
        print("Training Complete!")

if __name__ == "__main__":
    trainer = ASSCTrainer()
    trainer.train(epochs=5)
