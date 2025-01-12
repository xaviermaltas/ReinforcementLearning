STEPS = 30_000
env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = False,target_desire_factor=1)
eval_env = CartPoleEnvRandomTarget(render_mode=None,reward_function = 'custom',increased_actions = False,target_desire_factor=1,is_eval = True)

#TODO: Repetir el mateix que en els exercicis anteriors. Callback, model, avaluació inicial, càrrega del millor model i entrenament.
#Recordeu fer servir els millors hiperparametres trobats
#Al final, analitzar els resultats.