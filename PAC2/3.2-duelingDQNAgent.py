class duelingDQNAgent:
    def __init__(self, env, main_network, buffer, reward_threshold, epsilon=0.1, eps_decay=0.99, batch_size=32, device=None):
        """""
        Paràmetres
        ==========
        env: entorn
        target_network: classe amb la xarxa neuronal dissenyada
        target_network: xarxa objectiu
        buffer: classe amb el buffer de repetició d'experiències
        epsilon: epsilon
        eps_decay: decaïment d'epsilon
        batch_size: mida del batch
        nblock: bloc dels X últims episodis dels quals es calcularà la mitjana de recompensa
        reward_threshold: llindar de recompensa definit en l'entorn
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        ###############################################################
        # Inicialització de variables
        self.env = env
        self.main_network = main_network
        self.target_network = copy.deepcopy(main_network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = 100
        self.reward_threshold = reward_threshold
        self.initialize()

    def initialize(self):
        # Variables d'estat i seguiment
        self.state0 = self.env.reset()[0]
        self.total_reward = 0
        self.step_count = 0
        self.rewards_history = []
        self.mean_rewards_history = []
        self.episode_epsilon = []
        self.update_loss = []

    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            qvals = self.main_network.get_qvals(self.state0)
            action = torch.max(qvals, dim=-1)[1].item()
            self.step_count += 1

        next_state, reward, done, truncated, _ = self.env.step(action)
        self.total_reward += reward

        self.buffer.append(self.state0, action, reward, done, next_state)

        if done or truncated:
            self.state0 = self.env.reset()[0]
            return True
        else:
            self.state0 = next_state
            return False

    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000, min_episodios=250):
        self.gamma = gamma

        # Fase de burn-in
        print("Omplint el buffer de repetició d'experiències...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        print("Entrenant...")
        while training:
            self.state0 = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            while not gamedone:
                gamedone = self.take_step(self.epsilon, mode='train')

                ##### Actualitzar la xarxa principal segons la freqüència establerta #######
                if self.step_count % dnn_update_frequency == 0 and len(self.buffer.replay_memory) >= self.batch_size:
                    self.update()

                # Sincronització de xarxes
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.main_network.state_dict())

                if gamedone:
                    episode += 1
                    ##################################################################
                    ########Emmagatzemar epsilon, training rewards i loss #######

                    ####
                    self.update_loss = []
                    # Emmagatzemar mètriques
                    self.rewards_history.append(self.total_reward)
                    self.episode_epsilon.append(self.epsilon)

                    # Calcular mitjana mòbil
                    mean_rewards = np.mean(self.rewards_history[-self.nblock:]) if len(self.rewards_history) >= self.nblock else np.mean(self.rewards_history)
                    self.mean_rewards_history.append(mean_rewards)

                    print("\rEpisodi {:d} Recompenses Mitjanes {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")

                    # Comprovar si s'ha arribat al màxim d'episodis
                    if episode >= max_episodes:
                        training = False
                        print('\nLímit d\'episodis assolit.')
                        break

                    # Finalitza si la mitjana de recompenses arriba al llindar fixat
                    if mean_rewards >= self.reward_threshold and episode >= min_episodios:
                        training = False
                        print('\nEntorn resolt en {} episodis!'.format(episode))
                        break

                    # Actualització d'epsilon
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)

    ## Càlcul de la pèrdua
    def calculate_loss(self, batch):
        # Separem les variables de l'experiència i les convertim a tensors
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(self.device).reshape(-1, 1)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(self.device)
        dones_t = torch.ByteTensor(dones).to(self.device)

        # Obtenim els valors de Q de la xarxa principal
        qvals = torch.gather(self.main_network.get_qvals(states).to(self.device), 1, actions_vals)

        # Obtenim els valors de Q de la xarxa objectiu
        next_actions = torch.max(self.main_network.get_qvals(next_states).to(self.device), dim=-1)[1]
        next_actions_vals = next_actions.reshape(-1, 1).to(self.device)
        target_qvals = self.target_network.get_qvals(next_states).to(self.device)
        qvals_next = torch.gather(target_qvals, 1, next_actions_vals).detach()

        qvals_next[dones_t.bool()] = 0

        # Calculem equació de Bellman
        expected_qvals = None
        # Funció de pèrdua
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1, 1))
        return loss

    def update(self):
        self.main_network.optimizer.zero_grad()  # Eliminem qualsevol gradient passat
        batch = self.buffer.sample_batch(batch_size=self.batch_size)  # Seleccionem un conjunt del buffer
        loss = self.calculate_loss(batch)  # Calculem la pèrdua
        loss.backward()  # Obtenim els gradients
        self.main_network.optimizer.step()  # Apliquem els gradients a la xarxa neuronal
        # Emmagatzemem els valors de pèrdua
        if self.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
