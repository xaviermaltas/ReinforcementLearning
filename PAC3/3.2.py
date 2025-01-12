#implementar abans els TODO de l'entorn
#env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = True,target_desire_factor=1)
#eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = True,target_desire_factor=1,is_eval = True)

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Al final, analitzar els resultats.

#implementar abans els TODO de l'entorn
# env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = True,target_desire_factor=1)
# eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = True,target_desire_factor=1,is_eval = True)

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Al final, analitzar els resultats.
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Constants
STEPS = 20000
EVAL_FREQ = 2000
EVAL_EPISODES = 18

# Crear l'entorn amb les opcions especificades
env = CartPoleEnvRandomTarget(render_mode=None, reward_function='custom', increased_actions=True, target_desire_factor=1)
eval_env = CartPoleEnvRandomTarget(render_mode=None, reward_function='custom', increased_actions=True, target_desire_factor=1, is_eval=True)

# Callback for evaluation
callback = EvalCallback(
    eval_env=DummyVecEnv([lambda: eval_env]),
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=EVAL_FREQ, 
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=False
)

# Define PPO model
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_cartpole_tensorboard/"
)

# Avaluació inicial
initial_eval = evaluate_policy(
    model,
    DummyVecEnv([lambda: eval_env]),
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True
)
print(f"Initial evaluation: {initial_eval}")
print(f"Avaluació inicial - Mitjana de recompensa: {initial_eval[0]}")

# Entrenament
model.learn(total_timesteps=STEPS, callback=callback)

# Càrrega del millor model desat
best_model_path = "./best_model/best_model.zip"

if os.path.exists(best_model_path):
    model = PPO.load(best_model_path)
    print("El millor model ha estat carregat correctament.")
else:
    raise FileNotFoundError(f"No s'ha trobat el model a {best_model_path}")

# Avaluació final del millor model
final_eval = evaluate_policy(
    model,
    DummyVecEnv([lambda: eval_env]),
    n_eval_episodes=EVAL_EPISODES,
    return_episode_rewards=True
)

# Càlcul de la recompensa mitjana i passos mitjans
total_reward = sum(final_eval[0])  # Suma de totes les recompenses
total_steps = sum(final_eval[1])   # Suma de tots els passos

average_reward = total_reward / EVAL_EPISODES
average_steps = total_steps / EVAL_EPISODES
print(f'Best model evaluation: {final_eval}')
print(f"Recompensa mitjana: {average_reward}, Passos mitjans per episodi: {average_steps}")
