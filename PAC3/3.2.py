#implementar abans els TODO de l'entorn
env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = True,target_desire_factor=1)
eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = True,target_desire_factor=1,is_eval = True)

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Al final, analitzar els resultats.