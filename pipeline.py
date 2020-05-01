from dataclasses import dataclass
from utils import batch_iter
import torch
from tqdm import tqdm

@dataclass
class TrainArgs:
    device: str = 'cpu'
    learning_rate: float = 0.001
    epochs: int = 10
    train_batch_size: int = 32


class Pipeline:

    def __init__(self, dataloader, forward_model, loss_cls):
        self.dataloader = dataloader
        self.forward_model = forward_model
        self.criterion = loss_cls()

    def train_model(self, optimizer_cls, train_args=TrainArgs()):
        self.forward_model.train()
        self.forward_model.to(train_args.device)
        self.optimizer = optimizer_cls(self.forward_model.parameters(), lr=train_args.learning_rate)
        epoch = 0
        for epoch in range(train_args.epochs):
            running_loss = 0.0
            for _, (sent, mask, scores) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                outputs = self.forward_model.forward((sent.to(device=train_args.device), mask.to(device=train_args.device)))
                loss = self.criterion(outputs, scores.to(device=train_args.device))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print('epoch', epoch + 1, 'running loss', running_loss) 

    def evaluate(self, dataloader, batch_size=None):
        num_correct = 0 
        num_true_pos = 0
        num_pred_pos = 0
        num_true_and_pred_pos = 0
        with torch.no_grad():
            for i, (sent, mask, scores) in enumerate(dataloader):
                num_true_pos += scores.sum().item()
                _, predicted = torch.max(torch.exp(self.forward_model((sent, mask))), dim=1)
                num_pred_pos += predicted.sum().item()
                num_correct += (predicted == scores).sum().item()
                num_true_and_pred_pos += torch.min(predicted == scores, predicted == torch.ones(scores.shape[0])).sum().item()
        print('accuracy', num_correct / len(dataloader.dataset))
        print('recall', num_true_and_pred_pos / num_true_pos)
        print('precision', num_true_and_pred_pos / num_pred_pos)
        print('total pos', num_true_pos, 'total', len(dataloader.dataset))




