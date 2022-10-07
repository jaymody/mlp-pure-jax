# MLP in Jax
A pure jax implementation of a multi-layer perceptron neural network, trained on mnist.

The purpose of this repo is educational. While there are many open source examples of neural networks in Jax, many of them use supplementary frameworks like [haiku](https://github.com/deepmind/dm-haiku) or [flax](https://github.com/google/flax). My aim with this repo is to provide an example of neural networks in **JUST** jax that is simple to understand.

For comparison purposes, there is also `train_torch.py` which is the equivalent implementation of `train_jax` but in pytorch.

Both scripts run on cpu. `train_torch` can be trained on gpus by changing the `device` argument, `train_jax` can be trained on gpus by installing the jax with `cuda` extras (by default, `pyproject.toml` uses `cpu` extras).

### Usage
**Install Dependencies:**
```bash
poetry install
```
Test on `python 3.9.10`.

**Download MNIST Dataset:**
```bash
poetry run python data/download_mnist_as_csv.py
```
Requires `wget` and `gzip`.

**Train Neural Network with Jax:**
```bash
poetry run python train_jax.py
```

**Train Neural Network with PyTorch:**
```bash
poetry run python train_torch.py
```
