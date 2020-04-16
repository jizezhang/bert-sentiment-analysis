from dataclasses import dataclass
from utils import batch_iter
import torch

@dataclass
class TrainArgs:
    device: str = 'cpu'
    learning_rate: float = 0.001
    epochs: int = 10
    train_batch_size: int = 32


class Pipeline:

    def __init__(self, train_corpus, forward_model, loss_cls):
        self.train_corpus = train_corpus
        self.forward_model = forward_model
        self.criterion = loss_cls()

    def train_model(self, optimizer_cls, train_args=TrainArgs()):
        self.forward_model.train()
        self.forward_model.to(train_args.device)
        self.optimizer = optimizer_cls(self.forward_model.parameters(), lr=train_args.learning_rate)
        epoch = 0
        for epoch in range(train_args.epochs):
            running_loss = 0.0
            for sents, scores in batch_iter(self.train_corpus, train_args.train_batch_size, shuffle=True):
                self.optimizer.zero_grad()
                outputs = self.forward_model.forward(sents)
                loss = self.criterion(outputs, scores)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print('epoch', epoch, 'running loss', running_loss) 

    def evaluate(self, test_corpus, batch_size=None):
        num_correct = 0 
        with torch.no_grad():
            for sents, scores in batch_iter(test_corpus, batch_size):
                _, predicted = torch.max(self.forward_model(sents), 1)
                num_correct += (predicted == scores).sum().item()
        print('accuracy', num_correct / len(test_corpus))
        return num_correct




