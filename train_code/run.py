import torch
from torch_geometric.loader import DataLoader


if __name__ == "__main__":
    data_list = torch.load("train/game_2.pt")
    train_loader = DataLoader(data_list, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)
