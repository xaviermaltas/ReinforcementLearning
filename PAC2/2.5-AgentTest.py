import random
import pandas as pd
import numpy as np
# Generar un fitxer CSV que registri els resultats de les interaccions
# de l'agent amb el mercat en cada episodi i mostra per pantalla les últimes 30 accions.
file_path = 'stock_trading_agent_dqn.csv'

# Reproduir una partida completa de l'agent entrenat i mostrar el resultat final,
# incloent-hi el valor total del portafoli al final de l'episodi.
env = StockMarketEnv(
    ticker='SPY',
    start='2015-01-01',
    end='2024-01-01',
    is_eval=True,
    save_to_csv=True,
    csv_filename=file_path
)

def read_csv_and_show_last_30(file_path):
    try:
        # Llegir el fitxer CSV
        df = pd.read_csv(file_path)

        # Verificar que hi ha dades suficients
        if len(df) < 30:
            print(f"Només hi ha {len(df)} registres. Mostrant tots:")
            print(df)
        else:
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
    interaction_data = []

    for i_episode in range(505):
        # Generar nuevas fechas de inicio y fin aleatorias que cumplan con los días de trading deseados
        start_date, end_date = generate_random_trading_dates(
            start_range, 
            end_range, 
            trading_days_target
        )

        # Actualizar el entorno con las nuevas fechas
        env = base_env(
            ticker='SPY',
            start=start_date,
            end=end_date,
            is_eval=True,
            save_to_csv=True,
            csv_filename=f'episode_{i_episode+1}_actions.csv'
        )

        state = env.reset()[0]
        total_reward = 0
        win_days = 0
        episode_data = []

        while True:
            # El agente toma una acción
            # action = agent.get_action(state)
            action = agent.main_network.get_action(state)
            next_state, reward, done, trucated, _ = env.step(action)

            # Registrar dades de la interacció
            episode_data.append({
                'episode': i_episode + 1,
                'step': step + 1,
                'date': env.df.index[env.current_step].strftime('%Y-%m-%d'),
                'action': ['Mantenir', 'Comprar', 'Vendre'][action],
                'price': env.prices[env.current_step],
                'shares': env.shares_held,
                'balance': env.balance,
                'portfolio_value': env.net_worth,
                'reward': reward
            })

            # Actualizar recompensas y días positivos
            total_reward += reward
            if reward > 0:
                win_days += 1

            state = next_state

            if done or trucated:
                break

        all_rewards.append(total_reward)
        if win_days >= win_days_target:
            success_count += 1

        # Afegir dades de l'episodi al registre global
        interaction_data.extend(episode_data)

        env.close()

    success_rate = success_count / 100
    return all_rewards, success_rate

def plot_test(rewards, th):
    """
    Grafica los resultados del test.
    - rewards: Lista de recompensas totales por episodio
    - th: Umbral de recompensa establecido
    """

    plt.figure(figsize=(14, 7))
    
    #Gràfic principal
    plt.plot(rewards, alpha=0.6, color='steelblue', label='Dies Positius per Episodi')
    
    #Linia de tendència
    z = np.polyfit(range(len(rewards)), rewards, 1)
    p = np.poly1d(z)
    plt.plot(p(range(len(rewards)), "--", color='firebrick', label='Tendència')
    
    #Mitjana mòbil
    window_size = 30
    moving_avg = pd.Series(rewards).rolling(window_size).mean()
    plt.plot(moving_avg, color='darkorange', label=f'Mitjana Mòbil ({window_size} episodis)')
    
    #Línia de llindar
    plt.axhline(y=th, color='green', linestyle='--', linewidth=2, label='Objectiu de Rendiment')
    
    # Configuració del gràfic
    plt.title('Rendiment de la Estratègia en Diferents Condicions de Mercat', fontsize=14)
    plt.xlabel('Número d\'Episodi', fontsize=12)
    plt.ylabel('Dies amb Guany', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Quadre estadístic
    stats_text = f"""Estadístics Clau:
    - Mitjana: {np.mean(rewards):.1f} dies
    - Màxim: {np.max(rewards)} dies
    - Mínim: {np.min(rewards)} dies
    - Desviació Estàndard: {np.std(rewards):.1f}"""
    plt.gcf().text(0.72, 0.6, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
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