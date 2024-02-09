import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import TextCNN
from dataset import FoodDataset


def train(model, data_loader, loss_fn, optimizer, epoch):
    total_loss = 0
    for inputs, targets in tqdm(data_loader):
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch={epoch+1} loss={(total_loss / len(data_loader)):.4f}")


loss_fn = nn.CrossEntropyLoss()
max_text_len = 128
food_dataset = FoodDataset(
    "./data/result.tsv", model_file="./tokenizer.model", max_text_len=max_text_len
)

train_data_loader = DataLoader(food_dataset, batch_size=32)
model = TextCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load("checkpoint_4.pth"))

for epoch in range(10):
    epoch += 5

    train(model, train_data_loader, optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)
    torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")
