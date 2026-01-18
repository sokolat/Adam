import argparse

import torch
import torchvision
from torch import nn
from torchvision import transforms
from tqdm import tqdm


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

        for param in self.state["params"]:
            self.state["m"].append(torch.zeros_like(param))
            self.state["v"].append(torch.zeros_like(param))

    def step(self):
        self.state["timestep"] = self.state["timestep"] + 1
        for index, param in enumerate(self.state["params"]):
            if param.grad is not None:
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
                    torch.sqrt(self.state["v"][index]) + self.state["eps"]
                )


class LogisticRegression(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x


def main():

    model = LogisticRegression()
    optimizer = Adam(params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    batch_size = 128
    total_timesteps = 45

    train_data = torchvision.datasets.MNIST(
        root="./", train=True, transform=transforms.ToTensor(), download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    data_iter = iter(train_loader)

    for timestep in tqdm(range(total_timesteps)):

        data, target = next(data_iter)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        breakpoint()


if __name__ == "__main__":
    main()
