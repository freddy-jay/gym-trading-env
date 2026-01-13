import pandas as pd
import numpy as np
import gymnasium as gym
import gym_trading_env
from gym_trading_env.environments import TradingEnv
import pytest

def test_trading_env_basic():
    # Generate dummy data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    df = pd.DataFrame({
        "open": np.random.uniform(20000, 30000, 100),
        "high": np.random.uniform(20000, 30000, 100),
        "low": np.random.uniform(20000, 30000, 100),
        "close": np.random.uniform(20000, 30000, 100),
        "Volume USD": np.random.uniform(1000, 10000, 100)
    }, index=dates)
    df.index.name = "date"
    
    # Feature engineering
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df.dropna(inplace=True)

    env = gym.make(
        "TradingEnv",
        name="TestEnv",
        df=df,
        windows=5,
        positions=[-1, 0, 1],
        initial_position=0,
        trading_fees=0.01/100,
        portfolio_initial_value=1000,
        max_episode_duration=50
    )

    observation, info = env.reset()
    assert observation.shape == (5, 6) # 5 windows, 4 features + 2 dynamic features (last position, real position)
    
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    
    assert not done
    assert not truncated
    assert isinstance(reward, (float, np.float64, np.float32))

def test_multi_dataset_env():
    # Generate dummy data for two datasets
    dates1 = pd.date_range(start="2023-01-01", periods=100, freq="H")
    df1 = pd.DataFrame({
        "open": np.random.uniform(20000, 30000, 100),
        "high": np.random.uniform(20000, 30000, 100),
        "low": np.random.uniform(20000, 30000, 100),
        "close": np.random.uniform(20000, 30000, 100),
        "Volume USD": np.random.uniform(1000, 10000, 100)
    }, index=dates1)
    df1.index.name = "date"
    df1["feature_close"] = df1["close"].pct_change()
    df1.dropna(inplace=True)
    df1.to_pickle("tests/data1.pkl")

    dates2 = pd.date_range(start="2023-02-01", periods=100, freq="H")
    df2 = pd.DataFrame({
        "open": np.random.uniform(20000, 30000, 100),
        "high": np.random.uniform(20000, 30000, 100),
        "low": np.random.uniform(20000, 30000, 100),
        "close": np.random.uniform(20000, 30000, 100),
        "Volume USD": np.random.uniform(1000, 10000, 100)
    }, index=dates2)
    df2.index.name = "date"
    df2["feature_close"] = df2["close"].pct_change()
    df2.dropna(inplace=True)
    df2.to_pickle("tests/data2.pkl")

    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir="tests/*.pkl",
        positions=[-1, 0, 1],
        max_episode_duration=20
    )

    observation, info = env.reset()
    assert env.unwrapped.name in ["data1.pkl", "data2.pkl"]
    
    # Run an episode
    for _ in range(20):
        observation, reward, done, truncated, info = env.step(env.action_space.sample())
        if done or truncated:
            break
    
    import os
    os.remove("tests/data1.pkl")
    os.remove("tests/data2.pkl")
