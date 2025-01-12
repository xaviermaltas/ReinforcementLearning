from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
STEPS = 20_000
EVAL_FREQ = 2000
EVAL_EPISODES = 5

env = CartPoleEnvRandomTarget(render_mode=None, reward_function = 'default')

#best model save path
model_save_path = "./best_a2c_model"

#carregar callback
#callback = 
callback = EvalCallback(
    env,  # Environment for evaluation
    best_model_save_path=model_save_path,  # Directory to save the best model
    log_path=model_save_path,  # Log directory
    eval_freq=EVAL_FREQ,  # Evaluate the model every EVAL_FREQ steps
    n_eval_episodes=EVAL_EPISODES,  # Number of episodes for each evaluation
    deterministic=True,  # Use deterministic actions for evaluation
    render=False  # Don't render during evaluation
)

#carregar model
model = A2C('MlpPolicy', env, verbose=1)

#avaluació inicial
#initial_eval = 
initial_eval = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
print(f"Initial evaluation before training: Mean reward = {initial_eval[0]:.2f}, Std reward = {initial_eval[1]:.2f}")

#TODO: entrenar model
model.learn(total_timesteps=STEPS, callback=callback)

import os
#best model path
best_model_path = "./best_a2c_model/best_model.zip"

#carregar el millor model
if os.path.exists(best_model_path):
    model = A2C.load(best_model_path)
    print("El millor model ha estat carregat correctament.")
else:
    raise FileNotFoundError(f"No s'ha trobat el model a {best_model_path}")

#avaluació final
final_eval = final_eval = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
print (f'Initial evaluation: {initial_eval}')
print (f'Best model evaluation: {final_eval}')
print(f"Recompensa final mitjana: {final_eval[0]:.2f} +/- {final_eval[1]:.2f}")

#TODO: Completar funcio d'avaluació del model entrenat
def evaluate_trained_model(env,model):
    #Primer executem un cop l'entorn
    obs,_ = env.reset()
    done = False
    
    #Init variables
    observations = []
    rewards = []
    step = []

    while not done:
        # TODO: Escollir accio en base al model
        action, _ = model.predict(obs, deterministic=True)

        # TODO: Executar accio i esperar resposta de l'entorn
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # TODO: Guardar informacio necesaria per a poder fer les grafiques despres
        step.append((obs, action))
        rewards.append(reward)
        
        #update state and check end
        obs = next_obs
        done = terminated or truncated
        
    observations.append(step)

    #Analitzem resultats
    analyze_results(observations,rewards)
    print(f"Total Reward: {sum(rewards)}")

env = CartPoleEnvRandomTarget(render_mode='human',reward_function = 'default') #podeis hacer render mode = None
evaluate_trained_model(env,model)

env.close()