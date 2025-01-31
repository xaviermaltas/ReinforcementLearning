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
        ##### TODO 1: inicialitzar variables ######
        self.env = None  # TODO
        self.main_network = None  # TODO
        self.target_network = None  # TODO Xarxa objectiu (còpia de la principal)
        self.buffer = None  # TODO
        self.epsilon = None  # TODO
        self.eps_decay = None  # TODO
        self.batch_size = None  # TODO
        self.nblock = None  # TODO Bloc dels X últims episodis dels quals es calcularà la mitjana de recompensa
        self.reward_threshold = reward_threshold  # Llindar de recompensa definit en l'entorn

        self.initialize()

    ###############################################################
    ##### TODO 2: inicialitzar variables extres que es necessiten ######
    def initialize(self):
        pass
        # TODO

    #################################################################################
    ###### TODO 3: Prendre nova acció ###############################################
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            action = None  # TODO Acció aleatòria durant el burn-in
        else:
           action = None  # TODO Acció basada en el valor de Q (elecció de l'acció amb millor Q)
           self.step_count += 1

        # TODO: Realització de l'acció i obtenció del nou estat i la recompensa

        # TODO: reiniciar entorn 'if done'
        if done:
            pass  # TODO
        return done

    ## Entrenament
    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000, min_episodios=250):
        self.gamma = gamma
        # Omplim el buffer amb N experiències aleatòries
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
                # L'agent pren una acció
                gamedone = self.take_step(self.epsilon, mode='train')

                #################################################################################
                ##### TODO 4: Actualitzar la xarxa principal segons la freqüència establerta #######

                ########################################################################################
                ### TODO 6: Sincronitzar xarxa principal i xarxa objectiu segons la freqüència establerta #####

                if gamedone:
                    episode += 1
                    ##################################################################
                    ######## TODO: Emmagatzemar epsilon, training rewards i loss #######

                    ####
                    self.update_loss = []

                    #######################################################################################
                    ### TODO 7: Calcular la mitjana de recompensa dels últims X episodis i emmagatzemar #####
                    mean_rewards = None
                    ###

                    print("\rEpisodi {:d} Recompenses Mitjanes {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")

                    # Comprovar si s'ha arribat al màxim d'episodis
                    if episode >= max_episodes:
                        training = False
                        print('\nLímit d\'episodis assolit.')
                        break

                    # Finalitza si la mitjana de recompenses arriba al llindar fixat
                    if mean_rewards >= self.reward_threshold and min_episodios < episode:
                        training = False
                        print('\nEntorn resolt en {} episodis!'.format(episode))
                        break

                    #################################################################################
                    ###### TODO 8: Actualitzar epsilon ########
                    self.epsilon = None

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
