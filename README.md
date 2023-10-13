# Pokesim

## Overview

Pokesim provides:

-   A Unix server written in Node JS wrapped around the [sim](https://github.com/pkmn/pstree/main/sim) and [client](https://github.com/pkmn/ps/tree/main/client) packages from [pkmn](https://github.com/pkmn)
-   A reinforcement learning framework for interacting with this server. Currently supports [RNaD from deepmind](https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/python/algorithms/rnad).

## Installation

Run in terminal.

```bash
pip install -r requirements
```

Optionally, run this command for installing JAX gpu

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Running

In one terminal, run

```bash
make start
```

In one another, run

```bash
python main.py
```

## Configuration

Inside `./config.yml` you can change various things about how the algorithm is configured.
