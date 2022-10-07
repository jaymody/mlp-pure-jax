import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        sizes = [in_dim] + hidden_dims + [out_dim]
        self.fcs = nn.ModuleList(
            [nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        )

        # use same weight/bias initialization as train_jax.py
        for fc in self.fcs:
            fc.weight = nn.Parameter(nn.init.normal_(fc.weight.data))
            fc.bias = nn.Parameter(nn.init.zeros_(fc.bias.data))

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))

        last_fc = self.fcs[-1]
        return last_fc(x)


def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    for x, y in train_loader:
        # load data to device
        x = x.to(device).float()
        y = y.to(device)

        # feed forwards
        output = model(x)

        # compute loss
        loss = criterion(output, y)

        # backprop and step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def compute_loss_and_acc(model, data_loader, criterion, device):
    model.eval()
    running_loss, n_correct, n_total = 0, 0, 0
    for x, y in data_loader:
        # load data to device
        x = x.to(device).float()
        y = y.to(device)
        batch_size = x.shape[0]

        # feed forward
        with torch.no_grad():
            output = model(x)

        # compute loss
        loss = criterion(output, y)
        running_loss += loss.item() * batch_size

        # get predictions
        pred = torch.argmax(output, axis=-1)

        # add number of correct and total predictions made
        n_correct += torch.sum(pred == y).item()
        n_total += batch_size

    val_loss = running_loss / n_total
    val_acc = n_correct / n_total
    return val_loss, val_acc


def train(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    for epoch in range(n_epochs):
        train_step(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = compute_loss_and_acc(model, val_loader, criterion, device)
        train_loss, train_acc = compute_loss_and_acc(
            model, train_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1}/{n_epochs}\t"
            f"train_loss {train_loss:.3f}\t"
            f"train_acc {train_acc:.3f}\t"
            f"val_loss {val_loss:.3f}\t"
            f"val_acc {val_acc:.3f}\t"
        )


def train_mnist(
    train_filepath,
    test_filepath,
    train_batch_size,
    test_batch_size,
    lr,
    n_epochs,
    seed,
    device,
):
    from utils import read_mnist_csv, set_torch_seed

    # set seed for determinism
    set_torch_seed(seed)

    # load data
    train_X, train_y = read_mnist_csv(train_filepath)
    test_X, test_y = read_mnist_csv(test_filepath)

    train_X = train_X / 255.0
    test_X = test_X / 255.0

    # data loaders
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_X, train_y)),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_X, test_y)),
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # model
    model = MLP(784, [64], 10)
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # train loop
    train(model, train_loader, test_loader, criterion, optimizer, n_epochs, device)


if __name__ == "__main__":
    train_mnist(
        train_filepath="data/train.csv",
        test_filepath="data/test.csv",
        train_batch_size=64,
        test_batch_size=64,
        lr=1e-3,
        n_epochs=10,
        seed=123,
        device=torch.device("cpu"),
    )
