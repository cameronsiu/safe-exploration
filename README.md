# safe-exploration
Reimplemented https://github.com/AgrawalAmey/safe-explorer

## Requirements

- Follow the instructions to install uv (https://github.com/astral-sh/uv)
- Install Python 3.11: `uv python install 3.11`


## Installation

Inside the root directory:
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Training

By default, this command will train by default. Make sure to double check the `defaults.yml` file to also train the safety layer.
```
python3 -m safe_exploration.main
```

## Testing
```
python3 -m safe_exploration.main --main_trainer_test
```

## Isaac Sim

```
sudo chown -R $(id -u):$(id -g) ~/docker/isaac-sim
```

1. Open `people.usd`