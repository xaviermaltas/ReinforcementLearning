env = StockMarketEnv()

# Display the action space and observation space
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

import numpy as np

episodes = 100
total_reward = 0

for _ in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Select a random action
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
        
mean_reward = total_reward / episodes
print(f'\nMean reward over {episodes} episodes: {mean_reward}')

import time

state, _ = env.reset()
done = False

while not done:
    #env.render() #Uncomment to see all steps
    #time.sleep(0.1)
    action = env.action_space.sample()  #Random action
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.render()


import pandas as pd
def provar_save_to_csv(env, rows):
    """
    Provar la funció save_to_csv_file executant diverses accions
    a l'entorn i després mostrant les últimes 5 files del fitxer CSV.

    Paràmetres:
        env: L'entorn StockMarketEnv.
        rows: El número de files que es volen mostrar del fitxer CSV.
    """
    # Executar passos a l'entorn amb accions aleatòries
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Llegir i mostrar les últimes x files del fitxer CSV
    if env.save_to_csv:
        # Llegir el fitxer CSV en un DataFrame
        df = pd.read_csv(env.csv_filename)
        print(df.tail(rows))

env = StockMarketEnv(ticker="SPY", start="2019-01-01", end="2022-01-01", save_to_csv=True, csv_filename="stock_trading_log.csv")
provar_save_to_csv(env, 5)