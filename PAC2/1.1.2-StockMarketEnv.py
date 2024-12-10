import os
import csv

start = "2019-01-01"
end = "2021-01-01"
ticker = "SPY"
initial_balance = 10000

class StockMarketEnv(gym.Env):
    def __init__(self, ticker=ticker, initial_balance=initial_balance, is_eval=False,
                 start = start, end = end, save_to_csv=False,
                 csv_filename="stock_trading_log.csv"):
        super(StockMarketEnv, self).__init__()

        # Descarregar les dades històriques de l'acció
        self.df = yf.download(ticker, start=start, end=end)
        self.num_trading_days = len(self.df)
        self.prices = self.df['Close'].values
        self.n_steps = len(self.prices) - 1

        # Paràmetres de l'entorn
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = float(initial_balance)
        self.shares_held = 0
        self.net_worth = initial_balance
        self.previus_net_worth = initial_balance

        # Espai d'accions: 0 -> mantenir, 1 -> comprar, 2 -> vendre
        self.action_space = Discrete(3)

        # Calculem els indicadors tècnics
        self.rsi = calculate_rsi(self.df['Close']).values
        self.ema = calculate_ema(self.df['Close']).values

        # Espai d'observacions: [preu_actual, balanç, accions, rsi, ema]
        self.observation_space = Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        self.is_eval = is_eval

        # Valores para normalización (obtenemos mínimos y máximos)
        self.min_price = self.prices.min()
        self.max_price = self.prices.max()
        self.min_rsi = self.rsi.min()
        self.max_rsi = self.rsi.max()
        self.min_ema = self.ema.min()
        self.max_ema = self.ema.max()

        # Paràmetres addicionals per al fitxer CSV
        self.save_to_csv = save_to_csv
        self.csv_filename = csv_filename

        # Si l'opció de desar en CSV està activada, crea o sobreescriu el fitxer
        if self.save_to_csv:
            with open(self.csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Step', 'Balance', 'Shares Held', 'Net Worth', 'Profit'])

    def reset(self):
        """Reset the environment to its initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.previus_net_worth = self.initial_balance
        return self._next_observation(), {}

    def _normalize(self, value, min_val, max_val):
        """Normalize a value between 0 and 1."""
        if max_val == min_val:
            return 0.0
        normalized_value = (value - min_val) / (max_val - min_val)
        return float(normalized_value)

    def _next_observation(self):
        if self.current_step >= self.n_steps:
            return self._normalize(self.prices[-1], self.min_price, self.max_price), \
                   self._normalize(self.balance, self.initial_balance * 0.85, self.initial_balance * 1.25), \
                   self._normalize(self.shares_held, 0, self.initial_balance / self.prices[-1]), \
                   self._normalize(self.rsi[-1], self.min_rsi, self.max_rsi), \
                   self._normalize(self.ema[-1], self.min_ema, self.max_ema)
        
        # Normalitzem els valors
        norm_price = self._normalize(self.prices[self.current_step], self.min_price, self.max_price)
        norm_balance = self._normalize(self.balance, self.initial_balance * 0.85, self.initial_balance * 1.25)
        max_shares = self.initial_balance / self.prices[self.current_step]
        norm_shares_held = self._normalize(self.shares_held, 0, max_shares)
        norm_rsi = self._normalize(self.rsi[self.current_step], self.min_rsi, self.max_rsi)
        norm_ema = self._normalize(self.ema[self.current_step], self.min_ema, self.max_ema)

        return np.array([norm_price, norm_balance, norm_shares_held, norm_rsi, norm_ema])

    def step(self, action):
        """Execute one time step within the environment."""
        if self.current_step >= self.n_steps:
            terminated = True
            truncated = True
            return self._next_observation(), 0, terminated, truncated, {}

        current_price = self.prices[self.current_step]
        reward = 0

        # Acció: 0 -> mantenir, 1 -> comprar, 2 -> vendre
        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            self.balance = float(self.balance - shares_bought * current_price)
            self.shares_held += shares_bought
        elif action == 2:  # Sell
            self.balance = float(self.balance + self.shares_held * current_price)
            self.shares_held = 0

        # Actualitzar el preu anterior
        self.previus_net_worth = self.net_worth
        self.net_worth = float(self.balance + (self.shares_held * current_price))

        # Calcular la recompensa
        reward = self._calculate_reward()

        # Avançar al següent pas
        self.current_step += 1
        terminated = self.net_worth < (0.85 * self.initial_balance) or self.current_step >= self.n_steps
        truncated = self.current_step >= self.n_steps

        if self.save_to_csv:
            self.save_to_csv_file()

        # Retorna l'observació, la recompensa, si està complet, i altra informació addicional
        return self._next_observation(), reward, terminated, truncated, {}


    def _calculate_reward(self):
        reward = 0
        # primera etapa no computada
        if self.current_step == 0:
            return 0
        
        current_price = self.prices[self.current_step]
        previous_price = self.prices[self.current_step - 1]

        # net worth increased
        if self.net_worth > self.previus_net_worth:
            return 1
        # net worth decreased
        if self.net_worth < self.previus_net_worth:
            return -1
        if self.net_worth == self.previus_net_worth:
            # shares worth less than before
            if self.shares_held > 0 and current_price > previous_price:
                return 1
            # shares worth more than before
            if self.shares_held > 0 and current_price < previous_price:
                return -1
        return reward

    def render(self, mode='human'):
        """Render the environment state."""
        profit = float(self.net_worth - self.initial_balance)
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares held: {self.shares_held}")
        print(f"Net worth: {self.net_worth}")
        print(f"Profit: {profit}")

    def save_to_csv_file(self):
        """Desa les dades actuals al fitxer CSV."""
        profit = self.net_worth - self.initial_balance
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_step, self.balance, self.shares_held, self.net_worth, profit])
