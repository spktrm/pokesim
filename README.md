# Pokesim

## Overview

Pokesim provides:

-   A UNIX server written in Node JS wrapped around the [sim](https://github.com/pkmn/ps/tree/main/sim) and [client](https://github.com/pkmn/ps/tree/main/client) packages from [pkmn](https://github.com/pkmn)
-   A reinforcement learning framework for interacting with this server. Currently supports [R-NaD from deepmind](https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/python/algorithms/rnad), PPO and SAC.

## Installation

Run in terminal.

```bash
pip install -r requirements
sh ./scripts/download.sh
npm install
```

Optionally, run this command for installing JAX gpu

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Running

In one terminal, run the below command to start the UNIX server

```bash
make start
```

In one another, run the below command to begin training.

```bash
python main.py
```

## Configuration

Inside `./config.yml` you can change various things about how the algorithm is configured.
