
<h1 align='center'>
   <img src = 'https://github.com/ClementPerroud/Gym-Trading-Env/raw/main/docs/source/images/logo_light-bg.png' width='500'>
</h1>

<section class="shields" align="center">
   <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
         alt="python">
   </a>
   <a href="https://pypi.org/project/gym-trading-env/">
      <img src="https://img.shields.io/badge/pypi-v1.1.3-brightgreen.svg"
         alt="PyPI">
   </a>
   <a href="https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/LICENSE.txt">
   <img src="https://img.shields.io/badge/license-MIT%202.0%20Clause-green"
         alt="Apache 2.0 with Commons Clause">
   </a>
   <a href='https://gym-trading-env.readthedocs.io/en/latest/?badge=latest'>
         <img src='https://readthedocs.org/projects/gym-trading-env/badge/?version=latest' alt='Documentation Status' />
   </a>
   <a href="https://github.com/ClementPerroud/Gym-Trading-Env">
      <img src="https://img.shields.io/github/stars/ClementPerroud/gym-trading-env?style=social" alt="Github stars">
   </a>
</section>
  
Gym Trading Env is an Gymnasium environment for simulating stocks and training Reinforcement Learning (RL) trading agents.
It was designed to be fast and customizable for easy RL trading algorithms implementation.


| [Documentation](https://gym-trading-env.readthedocs.io/en/latest/index.html) |


Key features
---------------

This package aims to greatly simplify the research phase by offering :

* Easy and quick download technical data on several exchanges
* A simple and fast environment for the user and the AI, but which allows complex operations (Short, Margin trading).
* A high performance rendering (can display several hundred thousand candles simultaneously), customizable to visualize the actions of its agent and its results.
* (Coming soon) An easy way to backtest any RL-Agents or any kind 

![Render animated image](https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/docs/source/images/render.gif)

Installation
---------------

Gym Trading Env supports Python 3.9+ on Windows, Mac, and Linux. You can install it using pip:

```bash
pip install gym-trading-env
```

Or using git :

```bash
git clone https://github.com/ClementPerroud/Gym-Trading-Env
```

Minimal Example
---------------

```python
import gymnasium as gym
import gym_trading_env
import pandas as pd

# 1. Prepare your data
# The DataFrame must contain 'open', 'high', 'low', 'close' columns.
df = pd.read_csv("https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)

# 2. Design features
# The column names of your features must contain the keyword 'feature'.
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"]/df["close"]
df.dropna(inplace= True)

# 3. Create the environment
env = gym.make('TradingEnv', df=df, positions=[-1, 0, 1], portfolio_initial_value=1000)

# 4. Run the environment
observation, info = env.reset()
done, truncated = False, False
while not (done or truncated):
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
```


[Documentation available here](https://gym-trading-env.readthedocs.io/en/latest/index.html)
-----------------------------------------------------------------------------------------------
