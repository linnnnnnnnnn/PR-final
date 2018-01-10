# PR-final


## Table of Content

- Requirements
- Installation
- Module Design


## Requirements

- Linux or OSX
- python 2.7.x

## Installation

```bash
pip install -r requirements.txt
```

## Module Design

`models` to save humans' experience we've collected

`result` experiments results

### Network implementation (Deep Q network)

- `DQN_RL_brain.py` Natural DQN
- `DDQN_RL_brain.py` Double DQN 


### collector

- `car_collector.py` 
- `maze_collector.py`

### experiments

- `car_DDQN_test.py`
- `car_DQN_test.py`
- `maze_DQN_test.py`
- `maze_DDQN_test.py`

### demonstration
- `demo.py`


