env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom_1',target_desire_factor=0)
eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom_1',target_desire_factor=0,is_eval = True)

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Al final, analitzar els resultats.
#Repetir per a les dues noves funcions d'error.