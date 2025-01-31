import random
import pandas as pd
import numpy as np
# Generar un fitxer CSV que registri els resultats de les interaccions
# de l'agent amb el mercat en cada episodi i mostra per pantalla les últimes 30 accions.
file_path = 'stock_trading_agent_dqn.csv'

# Reproduir una partida completa de l'agent entrenat i mostrar el resultat final,
# incloent-hi el valor total del portafoli al final de l'episodi.
env = ''  # TODO: Assignar l'entorn entrenat

def read_csv_and_show_last_30(file_path):
    try:
        # Llegir el fitxer CSV
        df = pd.read_csv(file_path)

        # Mostrar les últimes 30 accions
        print("Últimes 30 accions del fitxer CSV:")
        print(df.tail(30))
    except FileNotFoundError:
        print(f"El fitxer {file_path} no s'ha trobat.")
    except Exception as e:
        print(f"S'ha produït un error en llegir el fitxer: {e}")

# Exemple d'ús
read_csv_and_show_last_30(file_path)


# Generar calendario de trading usando días hábiles
def generate_random_trading_dates(start_range, end_range, trading_days_target=505):
    """
    Genera un par de fechas (start, end) que tengan exactamente trading_days_target días hábiles.
    """
    start_date = pd.to_datetime(start_range)
    end_date = pd.to_datetime(end_range)

    while True:
        # Seleccionar una fecha de inicio aleatoria
        random_start = start_date + pd.DateOffset(days=random.randint(0, (end_date - start_date).days - trading_days_target))

        # Generar un rango de fechas de trading usando solo los días hábiles
        trading_days = pd.bdate_range(random_start, random_start + pd.DateOffset(days=2 * trading_days_target)).tolist()

        # Filtrar las fechas para obtener exactamente el número de días objetivo
        if len(trading_days) >= trading_days_target:
            random_end = trading_days[trading_days_target - 1]  # Último día de trading en el rango deseado
            return random_start.strftime("%Y-%m-%d"), random_end.strftime("%Y-%m-%d")

def test_model(agent, base_env, start_range, end_range, trading_days_target=505, win_days_target=252):
    """
    Testea el modelo entrenado en 100 episodios con fechas aleatorias de trading.

    Parámetros:
    - agent: Agente entrenado
    - base_env: Entorno base
    - start_range, end_range: Rangos de fechas para generar las fechas de trading
    - trading_days_target: Días de trading por episodio
    - win_days_target: Días positivos requeridos para considerar éxito en un episodio

    Retorna:
    - all_rewards: Lista con las recompensas totales de cada episodio
    - success_rate: Porcentaje de episodios exitosos
    """
    all_rewards = []
    success_count = 0

    for i_episode in range(100):
        # Generar nuevas fechas de inicio y fin aleatorias que cumplan con los días de trading deseados
        start_date, end_date = generate_random_trading_dates(start_range, end_range, trading_days_target)

        # Actualizar el entorno con las nuevas fechas
        env = base_env  # TODO: Inicializar entorno con las fechas generadas

        state = env.reset()
        total_reward = 0
        win_days = 0

        while True:
            # El agente toma una acción
            action = agent.get_action(state)  # TODO: Obtener la acción del agente
            next_state, reward, done, _ = env.step(action)

            # Actualizar recompensas y días positivos
            total_reward += reward
            if reward > 0:
                win_days += 1

            state = next_state

            if done:
                break

        all_rewards.append(total_reward)
        if win_days >= win_days_target:
            success_count += 1

        env.close()

    success_rate = success_count / 100
    return all_rewards, success_rate

def plot_test(rewards, th):
    """
    Grafica los resultados del test.
    - rewards: Lista de recompensas totales por episodio
    - th: Umbral de recompensa establecido
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Recompensas Totales")
    plt.axhline(y=th, color='r', linestyle='-', label="Umbral de Recompensa")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensas")
    plt.title("Resultados del Test del Modelo")
    plt.legend()
    plt.show()


# TODO: Calcular la recompensa mitjana de les 100 partides de test
mean_reward_dqn = np.mean(all_rewards)  # Calcular la mitjana de recompenses obtingudes en el test

# TODO: Assignar la Mean Reward de la última iteració de l'entrenament
mean_reward_dqn_last = mean_rewards[-1]  # Última recompensa mitjana obtinguda durant l'entrenament

# TODO: Calcular el percentatge d'episodis exitosos
success_rate = success_count / 100  # Proporció d'episodis amb més de 252 dies positius

# Resultats
print(f"La recompensa mitjana obtinguda per l'agent DQN en les 100 partides de test és: {mean_reward_dqn:.2f} punts.")
print(f"Percentatge d'episodis que van aconseguir guanyar almenys 252 dies: {success_rate * 100:.2f}%")

