# settemezzo

This repository contains a PIP package which is an OpenAI environment for simulating an enironment in which settemezzo is played.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```bash
cd gym-settemezzo
pip install -e .
```

## Usage

```
import gym
import gym_settemezzo

env = gym.make('Settemezzo-v0')
```

