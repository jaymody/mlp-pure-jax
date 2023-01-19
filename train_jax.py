from typing import NamedTuple

import jax
import jax.numpy as jnp


class Params(NamedTuple):
    w1: jnp.ndarray  # [d_in, d_hidden]
    b1: jnp.ndarray  # [d_hidden]
    w2: jnp.ndarray  # [d_hidden, d_out]
    b2: jnp.ndarray  # [d_out]


def init_params(d_in, d_hidden, d_out, key):
    w1_key, w2_key = jax.random.split(key)
    return Params(
        w1=jax.random.normal(key=w1_key, shape=[d_in, d_hidden]),
        b1=jnp.zeros(shape=[d_hidden]),
        w2=jax.random.normal(key=w2_key, shape=[d_hidden, d_out]),
        b2=jnp.zeros(shape=[d_out]),
    )


@jax.jit
def neural_network(params, x):
    x = x @ params.w1 + params.b1
    x = jax.nn.relu(x)
    x = x @ params.w2 + params.b2
    return x


@jax.jit
@jax.vmap
def cross_entropy_loss(logits, y):
    return -jax.nn.log_softmax(logits)[y]


@jax.jit
def loss_fn(params, x, y):
    logits = neural_network(params, x)
    loss = jnp.mean(cross_entropy_loss(logits, y))  # average loss across examples
    return loss


@jax.jit
def update(params, x, y, lr):
    grad = jax.grad(loss_fn)(params, x, y)
    params = jax.tree_map(lambda w, g: w - lr * g, params, grad)  # gradient descent
    return params


def get_batches(X, Y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], Y[i : i + batch_size]


def compute_metrics(params, X, Y, batch_size):
    loss, n_correct = 0, 0
    for x, y in get_batches(X, Y, batch_size):
        logits = neural_network(params, x)

        loss += jnp.sum(cross_entropy_loss(logits, y))

        preds = jnp.argmax(logits, axis=-1)
        n_correct += jnp.sum(preds == y)

    n_total = X.shape[0]
    return loss / n_total, n_correct / n_total


def train_mnist(
    train_filepath="data/train.csv",
    test_filepath="data/test.csv",
    batch_size=64,
    d_hidden=64,
    lr=1e-3,
    n_epochs=10,
    seed=123,
):
    from utils import read_mnist_csv

    # load data
    train_X, train_Y = read_mnist_csv(train_filepath)
    val_X, val_Y = read_mnist_csv(test_filepath)

    # normalize inputs to float numbers between 0 and 1 (inclusive)
    train_X = train_X / 255.0
    val_X = val_X / 255.0

    # initialize parameters
    params = init_params(784, d_hidden, 10, key=jax.random.PRNGKey(seed))

    # train loop
    for epoch in range(n_epochs):
        for x, y in get_batches(train_X, train_Y, batch_size):
            params = update(params, x, y, lr)

        train_loss, train_acc = compute_metrics(params, train_X, train_Y, batch_size)
        val_loss, val_acc = compute_metrics(params, val_X, val_Y, batch_size)
        print(
            f"Epoch {f'{epoch+1}/{n_epochs}':<10}"
            f"train_loss {train_loss:<10.3f}"
            f"train_acc {train_acc:<10.3f}"
            f"val_loss {val_loss:<10.3f}"
            f"val_acc {val_acc:<10.3f}"
        )


if __name__ == "__main__":
    train_mnist()
