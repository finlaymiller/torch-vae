import sys
import os
import unittest
import torch
from rich import print
from torchinfo import summary

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vanilla_vae import VanillaVAE


class TestVAE(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")
        self.model = VanillaVAE(3, 10)
        self.model.to(self.device)

    def test_summary(self):
        print(summary(self.model, input_size=(64, 3, 64, 64), device="cuda"))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64).to(self.device)
        y = self.model(x)
        print("Model Output size:", y["output"].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64).to(self.device)
        result = self.model(x)
        loss = self.model.loss(result, kld_weight=0.005)
        print(f"Loss is {loss}")


if __name__ == "__main__":
    unittest.main()
