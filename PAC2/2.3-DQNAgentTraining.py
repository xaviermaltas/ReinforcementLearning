# Define variables
learning_rate = 0.0005
batch_size = 128
max_episodes = 4000
BURN_IN = 1000
dnn_update_frequency = 6
dnn_sync_frequency = 15
MEMORY_SIZE = 50000
gamma = 0.99
epsilon = 1.0  # inicialitzar epsilon
eps_decay = 0.995
ticker = 'SPY'
start = '2023-01-01' 
end = '2024-01-01'

# Convertir les dates a `datetime`
from datetime import datetime
start_date = datetime.strptime(start, '%Y-%m-%d')
end_date = datetime.strptime(end, '%Y-%m-%d')

# Calcular el nombre de dies de trading entre les dates
num_days = (end_date - start_date).days

print(f"Nombre de dies de trading per {ticker} des de {start} fins a {end}: {num_days}")
print(f"El nostre objectiu és guanyar el 50% dels dies: {round(num_days / 2)}")

# Calcular el REWARD_THRESHOLD
REWARD_THRESHOLD = round(num_days / 2)
print(f"REWARD_THRESHOLD calculat: {REWARD_THRESHOLD}")

# TODO: Entrenament
# Temps d'entrenament a Google Colab amb GPU: 42.53 minuts
# De mitjana obté entre 170-190 de puntuació i arriba als 4000 episodis.

# Importar les llibreries necessàries
import torch
import torch.nn as nn  # Assegura't que aquest mòdul estigui importat
from datetime import datetime

# Crear l'entorn i l'agent
env = StockMarketEnv(ticker, initial_balance=10000.0, is_eval=False, start=start_date, end=end_date, save_to_csv=False, csv_filename='')  # assegura't que l'entorn estigui correctament definit
network = NeuralNetStockMarket(env, learning_rate=learning_rate)
buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
agent = DQNAgent(env, network, buffer, epsilon=epsilon, eps_decay=eps_decay, batch_size=batch_size, min_episodes=300)

# Iniciar el procés d'entrenament
agent.train(gamma=gamma, max_episodes=max_episodes, batch_size=batch_size, dnn_update_frequency=dnn_update_frequency, dnn_sync_frequency=dnn_sync_frequency, REWARD_THRESHOLD=REWARD_THRESHOLD)