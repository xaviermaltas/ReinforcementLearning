# TODO: Temps d'execució 69 minuts a Google Colaboratory amb GPU.
# Resultat esperat al voltant de 180-200 punts.

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
min_episodes = 300
ticker = 'SPY'
start = '2020-01-01'
end = '2023-01-01'

# Calcular el nombre de dies de trading entre les dates
num_days = (end_date - start_date).days

print(f"Nombre de dies de trading per {ticker} des de {start} fins a {end}: {num_days}")
print(f"El nostre objectiu és guanyar el 50% dels dies: {round(num_days / 2)}")

# Calcular el REWARD_THRESHOLD
REWARD_THRESHOLD = round(num_days / 2)
print(f"REWARD_THRESHOLD calculat: {REWARD_THRESHOLD}")

env = StockMarketEnv(ticker=ticker, start=start, end=end)
buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
main_network = duelingDQN(env, learning_rate=learning_rate)

# Create DQN agent
agent = duelingDQNAgent(
    env=env,
    main_network=main_network,
    buffer=buffer,
    reward_threshold=REWARD_THRESHOLD,
    epsilon=epsilon,
    eps_decay=eps_decay,
    batch_size=batch_size
)

# Start training
agent.train(
    gamma=gamma,
    max_episodes=max_episodes,
    batch_size=batch_size,
    dnn_update_frequency=dnn_update_frequency,
    dnn_sync_frequency=dnn_sync_frequency,
    min_episodios=min_episodes
)