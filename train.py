from dataclasses import dataclass
from utils import batch_iter

@dataclass
class TrainArgs:
    device: str = 'cpu'
    learning_rate: float = 0.001
    epoch: int = 10
    train_batch_size: int = 32


def train_model(corpus, forward_model, optimizer_cls, loss_cls, train_args=TrainArgs()):
    forward_model.train()
    forward_model.to(train_args.device)
    optimizer = optimizer_cls(forward_model.parameters(), lr=train_args.learning_rate)
    criterion = loss_cls()
    epoch = 0
    for epoch in range(train_args.epoch):
        running_loss = 0.0
        for sents, scores in batch_iter(corpus, train_args.train_batch_size, shuffle=True):
            optimizer.zero_grad()
            outputs = forward_model.forward(sents)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('epoch', epoch, 'running loss', running_loss)
    return forward_model



