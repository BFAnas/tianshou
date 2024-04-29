import os
import datetime
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import gymnasium as gym

from examples.offline.utils import load_buffer_d4rl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Novelty Detection")
    parser.add_argument("--task", default="Hopper-v2", type=str, help="Task name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-data", default="hopper-medium-v2", type=str, help="Task data")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use for computation")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate for the model")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size for training")
    parser.add_argument("--num-repeat", default=10, type=int, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--repeat-ID", default=False, action="store_true")
    parser.add_argument("--hidden-sizes", nargs='+', default=[1024, 512, 256], type=int, help="Hidden layer sizes for the model")
    parser.add_argument("--logdir", type=str, default="log")
    return parser.parse_args()

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            for i in range(len(hidden_sizes) - 1)
        ])
        self.hidden_norms = nn.ModuleList([
            nn.LayerNorm(hidden_sizes[i+1])
            for i in range(len(hidden_sizes) - 1)
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)  # 1 output for energy score
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = torch.relu(x)
        for hidden_layer, hidden_norm in zip(self.hidden_layers, self.hidden_norms):
            x = hidden_layer(x)
            x = hidden_norm(x)
            x = torch.relu(x)
        output = self.output_layer(x)
        return torch.sigmoid(output)
    
class MyData(Dataset):
    def __init__(self, ID, OD, device):
        self.ID = torch.from_numpy(ID)
        self.OD = torch.from_numpy(OD)
        self.X = torch.concatenate((self.ID, self.OD)).to(device)
        self.y = torch.concatenate((torch.ones(len(self.ID)), torch.zeros(len(self.OD)))).unsqueeze(-1).to(device)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

def train_loop(dataloader, model, loss_fn, optimizer, save_path, val_dataloader=None, epochs=1, writer=None):
    min_val_loss = np.infty
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()  # Zero gradients before forward pass
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataloader)

        if val_dataloader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_dataloader:
                    pred_val = model(X_val)
                    val_loss = loss_fn(pred_val, y_val).item()
                    total_val_loss += val_loss

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            if writer:
                writer.add_scalar("Loss/train", avg_train_loss, epoch)
                writer.add_scalar("Loss/val", avg_val_loss, epoch)
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

def test_model(test_dataloader, model, writer):
    with torch.no_grad():
        test_pred = model(test_dataloader.dataset.X).round().cpu()
        test_labels = test_dataloader.dataset.y.cpu()

    cm = confusion_matrix(test_labels, test_pred)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    writer.add_text("Confusion Matrix", str(cm))

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    IDbuffer = load_buffer_d4rl(args.task_data)
    if args.repeat_ID:
        IDdata = np.concatenate((np.repeat(IDbuffer.obs, args.num_repeat, axis=0), np.repeat(IDbuffer.act, args.num_repeat, axis=0)), axis=1)
    else:
        IDdata = np.concatenate((IDbuffer.obs, IDbuffer.act), axis=1)
    env = gym.make(args.task)
    num_samples = args.num_repeat*len(IDbuffer)
    rand_actions = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=(num_samples, env.action_space.shape[0])).astype(np.float32)
    ODdata = np.concatenate((np.repeat(IDbuffer.obs, args.num_repeat, axis=0), rand_actions), axis=1)

    train_ID, test_ID = train_test_split(IDdata, test_size=0.2)
    train_ID, val_ID = train_test_split(train_ID, test_size=0.2)
    train_OD, test_OD = train_test_split(ODdata, test_size=0.2)
    train_OD, val_OD = train_test_split(train_OD, test_size=0.2)

    train_data = MyData(train_ID, train_OD, args.device)
    val_data = MyData(val_ID, val_OD, args.device)
    test_data = MyData(test_ID, test_OD, args.device)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    train_features, _ = next(iter(train_dataloader))
    input_size = list(train_features[0].shape)[0]

    model = MyModel(input_size, args.hidden_sizes).to(args.device)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(args.task, "action_classification", str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    train_loop(train_dataloader, model, loss_fn, optimizer, save_path=log_path, val_dataloader=val_dataloader, epochs=args.epoch, writer=writer)
    test_model(test_dataloader, model, writer)

if __name__ == "__main__":
    main()