#########################
# Model training
#########################
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv

#Modificar primer els TODOS de l'entorn
EVAL_EPISODES = 18
STEPS = 20000
EVAL_FREQ = 2000
env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'default')
eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'default',is_eval = True)

#TODO: Completar codi
#Creem callback i model. Per al callback farem servir l'entorn d'evaluacio, per al model el normal
callback = EvalCallback(
    eval_env=DummyVecEnv([lambda: eval_env]),  # Vectoritzar l'entorn d'avaluació
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=EVAL_FREQ,  # Freqüència d'avaluació
    n_eval_episodes=EVAL_EPISODES,  # Episodis d'avaluació
    deterministic=True,
    render=False
)

model = PPO(
    "MlpPolicy",  # Política basada en xarxes neuronals
    env,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

#Executem evaluacio inicial
initial_eval = evaluate_policy(
    model,
    DummyVecEnv([lambda: eval_env]),
    n_eval_episodes=9,  # Un episodi per a cada casuística
    return_episode_rewards=True  # Retornar els resultats per episodi
)
print(f"Initial evaluation: {initial_eval}")

#Entrenem model
model.learn(total_timesteps=STEPS, callback=callback)

#Carreguem millor model i executem validació final
best_model_path = "./best_model/best_model.zip"
if os.path.exists(best_model_path):
    model = PPO.load(best_model_path)
    print("El millor model ha estat carregat correctament.")
else:
    raise FileNotFoundError(f"No s'ha trobat el model a {best_model_path}")

final_eval = evaluate_policy(
    model,
    DummyVecEnv([lambda: eval_env]),
    n_eval_episodes=9,  # Un episodi per a cada casuística
    return_episode_rewards=True
)
# print (f'Initial evaluation: {initial_eval}')
print (f'Best model evaluation: {final_eval}')

#########################
# New analyze_results function
#########################

def analyze_results(observations, rewards):
    """
    Genera subplots per cada casuística:
    - Gràfiques de trajectòries "distància-angle"
    - Gràfiques d'acumulació de recompenses
    """
    n_cases = len(observations)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # Subplots de 3 files i 5 columnes
    axes = axes.flatten()  # Aplanem per facilitar l'iteració

    for case_idx in range(n_cases):
        obs_case = observations[case_idx]
        rewards_case = rewards[case_idx]

        # Dades d'observacions i accions
        positions = [obs[0][0] for obs in obs_case]  # Posicions relatives al target
        angles = [obs[0][2] for obs in obs_case]  # Angles del pal
        actions = [obs[1] for obs in obs_case]  # Accions

        # Gràfic de trajectòria posició-angle
        ax = axes[case_idx]
        ax.plot(positions, angles, label="Trajectòria", color='blue', alpha=0.8)
        ax.scatter(positions[0], angles[0], label="Inicial", color='green', s=50)
        ax.scatter(positions[-1], angles[-1], label="Final", color='red', s=50)

        ax.set_title(f"Casuística {case_idx + 1} (Recompensa: {sum(rewards_case)})")
        ax.set_xlabel("Distància al target")
        ax.set_ylabel("Angle del pal (rad)")
        ax.legend()
        ax.grid(alpha=0.3)

    # Omplir subplots sobrants (si n_cases < 15)
    for empty_ax in axes[n_cases:]:
        empty_ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Gràfiques d'acumulació de recompenses amb informació extra
    fig, reward_axes = plt.subplots(3, 5, figsize=(20, 12))
    reward_axes = reward_axes.flatten()

    for case_idx in range(n_cases):
        rewards_case = rewards[case_idx]
        cumulative_rewards = np.cumsum(rewards_case)

        # Gràfic d'acumulació de recompenses
        reward_ax = reward_axes[case_idx]
        reward_ax.plot(range(len(cumulative_rewards)), cumulative_rewards, label="Recompensa acumulada", color='orange')
        reward_ax.set_title(f"Casuística {case_idx + 1}")
        reward_ax.set_xlabel("Pas")
        reward_ax.set_ylabel("Recompensa acumulada")
        reward_ax.grid(alpha=0.3)

        # Afegir anotació amb el valor de recompensa final
        reward_ax.text(
            len(cumulative_rewards) - 1, cumulative_rewards[-1], 
            f"Final: {cumulative_rewards[-1]:.1f}", 
            color='red', fontsize=10, ha='right'
        )

        # Mostrar recompensa en cada step a la gràfica
        for step, reward in enumerate(rewards_case):
            reward_ax.annotate(
                f"{reward:.1f}", 
                (step, cumulative_rewards[step]),
                textcoords="offset points", 
                xytext=(0, 5), 
                ha='center', fontsize=7, color='blue'
            )

    # Omplir subplots sobrants (si n_cases < 15)
    for empty_ax in reward_axes[n_cases:]:
        empty_ax.axis('off')

    plt.tight_layout()
    plt.show()

#########################
# evaluate_trained_model function
#########################

from tqdm import tqdm

def evaluate_trained_model(env, model):
    """
    Avaluem un model entrenat per 9 casuístiques i generem gràfiques.
    """
    n_eval_cases = 9
    observations = []  # Guardar les observacions (trajectòries)
    rewards = []  # Guardar les recompenses per cada casuística

    # Executem una iteració per casuística
    for _ in tqdm(range(n_eval_cases), desc="Avaluant casuístiques"):
        done = False
        episode_obs = []
        episode_rewards = []

        obs, _ = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Acció del model entrenat
            next_obs, reward, done, _, _ = env.step(action)  # Pas de simulació

            # Guardem observacions, accions i recompenses
            episode_obs.append((obs, action))
            episode_rewards.append(reward)
            obs = next_obs

        # Guardem trajectòria i recompenses d'aquest cas
        observations.append(episode_obs)
        rewards.append(episode_rewards)

    # Generem gràfiques amb la funció `analyze_results`
    analyze_results(observations, rewards)



#########################
# Evaluate trained model 
#########################

#Executem la funcio
eval_env = CartPoleEnvRandomTarget(render_mode='human',reward_function = 'default',is_eval = True) #podeis modificar el render mode a None
eval_env = CartPoleEnvRandomTarget(render_mode='None',reward_function = 'default',is_eval = True)
evaluate_trained_model(eval_env,model)

eval_env.close()