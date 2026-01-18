import argparse

import torch
from torch import nn


class Adam:
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8) -> None:
        self.state = {}
        self.state["lr"] = lr
        self.state["beta_1"] = beta_1
        self.state["beta_2"] = beta_2
        self.state["eps"] = eps
        self.state["params"] = list(params)
        self.state["timestep"] = 0
        self.state["m"] = []
        self.state["v"] = []

        for param in params:
            self.state["m"].append(torch.zeros_like(param))
            self.state["v"].append(torch.zeros_like(param))

    def step(self):
        self.state["timestep"] = self.state["timestep"] + 1
        for index, param in enumerate(self.state["params"]):
            if param.grad:
                self.state["m"][index] = (
                    self.state["beta_1"] * self.state["m"][index]
                    + (1 - self.state["beta_1"]) * param.grad
                )
                self.state["v"][index] = (
                    self.state["beta_2"] * self.state["v"][index]
                    + (1 - self.state["beta_2"]) * param.grad**2
                )
                self.state["m"][index] = self.state["m"][index] / (
                    1 - self.state["beta_1"] ** self.state["timestep"]
                )
                self.state["v"][index] = self.state["v"][index] / (
                    1 - self.state["beta_2"] ** self.state["timestep"]
                )
                param = param - self.state["lr"] * self.state["m"][index] / (
                    torch.sqrt(self.state["v"]) + self.state["eps"]
                )


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(20, 30)

    def forward(self, x):
        x = self.fc(x)
        return x


def main():
    model = Model()
    optimizer = Adam(params=model.parameters())
    optimizer.step()


if __name__ == "__main__":
    main()
