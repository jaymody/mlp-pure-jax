import jax
import jax.numpy as jnp


def forward_fn(params, X):
    for W, b in params[:-1]:
        X = jax.nn.relu(X @ W + b)

    final_W, final_b = params[-1]
    return X @ final_W + final_b


def initialize_params(key, input_dim, hidden_dims, output_dim):
    sizes = [input_dim] + hidden_dims + [output_dim]
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        (jax.random.normal(k, (n_in, n_out)), jnp.zeros((n_out,)))
        for k, n_in, n_out in zip(keys, sizes[:-1], sizes[1:])
    ]


def cross_entropy_loss(unnormalized_probs, y):
    batch_size = unnormalized_probs.shape[0]
    num_classes = unnormalized_probs.shape[-1]
    log_probs = jax.nn.log_softmax(unnormalized_probs, axis=-1)
    labels = jax.nn.one_hot(y, num_classes)
    loss = jnp.sum(labels * -log_probs) / batch_size
    return loss


def loss_fn(params, X, y):
    unnormalized_probs = forward_fn(params, X)
    loss = cross_entropy_loss(unnormalized_probs, y)
    return loss


@jax.jit
def update(params, X, y, lr):
    # compute gradient
    grad = jax.grad(loss_fn)(params, X, y)

    # good ole vanilla stochastic gradient descent
    params = jax.tree_map(lambda w, g: w - lr * g, params, grad)

    return params


def compute_loss_and_accuracy(params, data_X, data_y, batch_size):
    running_loss, n_correct, n_total = 0, 0, 0
    for X, y in get_batches(data_X, data_y, batch_size):
        batch_size = X.shape[0]

        # feed forward
        unnormalized_probs = forward_fn(params, X)

        # compute loss
        running_loss += cross_entropy_loss(unnormalized_probs, y) * batch_size

        # accuracy
        preds = jnp.argmax(unnormalized_probs, axis=-1)
        n_correct += jnp.sum(preds == y)
        n_total += batch_size

    loss = running_loss / n_total
    accuracy = n_correct / n_total
    return loss, accuracy


def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


def train(
    params,
    train_X,
    train_y,
    val_X,
    val_y,
    lr,
    n_epochs,
    train_batch_size,
    val_batch_size,
):
    for epoch in range(n_epochs):
        for X, y in get_batches(train_X, train_y, train_batch_size):
            params = update(params, X, y, lr)

        train_loss, train_acc = compute_loss_and_accuracy(
            params, train_X, train_y, train_batch_size
        )
        val_loss, val_acc = compute_loss_and_accuracy(
            params, val_X, val_y, val_batch_size
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
):
    from utils import read_mnist_csv

    train_X, train_y = read_mnist_csv(train_filepath)
    test_X, test_y = read_mnist_csv(test_filepath)

    train_X = train_X / 255.0
    test_X = test_X / 255.0

    key = jax.random.PRNGKey(seed)
    params = initialize_params(key, 784, [64], 10)

    train(
        params,
        train_X,
        train_y,
        test_X,
        test_y,
        lr,
        n_epochs,
        train_batch_size,
        test_batch_size,
    )


if __name__ == "__main__":
    train_mnist(
        train_filepath="data/train.csv",
        test_filepath="data/test.csv",
        train_batch_size=64,
        test_batch_size=64,
        lr=1e-3,
        n_epochs=10,
        seed=123,
    )
