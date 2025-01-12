"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled



class CartPoleEnvRandomTarget(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to a modification of the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.
     The modification consists on a target appearing into scene. The cart must be as close to the target as posible

    For more details about original cartpole look for source code in gymnasium.
    """


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        target_desire_factor: float = 0.5,
        reward_function: str = "default",
        is_eval=False,
        increased_actions=False,
        render_mode: Optional[str] = None,
    ):

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.max_steps = 500
        self.steps = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.target_threshold = 2

        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
                self.target_threshold * 2,
            ],
            dtype=np.float32,
        )
        self.increased_actions = increased_actions
        if self.increased_actions:
            #TODO 3.1: Ampliar l'espai d'observacions
            raise NotImplementedError("Increased actions should be implemented")
        else:
            self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

        #TODO 1.3: Tenir en compte si l'entorn és d'avaluació
        self.is_eval = is_eval
        self.target_position_index = None #index to control target position
        self.target_position = self.generate_random_target_position()

        self.target_desire_factor = min(
            max(target_desire_factor, 0), 1
        )  # between 0 and 1
        if reward_function in ["default", "custom", "custom1", "custom2"]:
            self.reward_function = reward_function
        else:
            raise AttributeError("reward function must be either default or custom")

    def generate_random_target_position(
        self,
    ):
        if self.is_eval:
            #TODO 1.3:Tingueu en compte si l'entorn és d'avaluació per situar el target en unes posicions concretes.
            #preset target positions for evaluation mode
            eval_positions = [
                -self.x_threshold,                      # Extrem esquerre
                -self.x_threshold * 0.75,               # 3/4 esquerre
                -self.x_threshold * 0.5,                # 1/2 esquerre
                -self.x_threshold * 0.25,               # 1/4 esquerre
                0.0,                                    # Centre
                self.x_threshold * 0.25,                # 1/4 dret
                self.x_threshold * 0.5,                 # 1/2 dret
                self.x_threshold * 0.75,                # 3/4 dret
                self.x_threshold                        # Extrem dret
            ]
            #select a random position (first execution)
            if self.target_position_index is None:
                self.target_position_index = np.random.randint(0,9)

            #select the target position based on the index
            target_position = eval_positions[self.target_position_index]

            #update the index
            self.target_position_index = (self.target_position_index + 1) % 9
            
            return target_position
        else:
            return np.random.uniform(-self.x_threshold, self.x_threshold)

    def custom_reward(self, target_position, current_position, angle, terminated):
        if self.reward_function == "default":
            #Reward default del cartpole
            return 1 if not terminated else 0
        elif self.reward_function == "custom":
            angle_reward = ( -abs(angle) / (2.0 * self.x_threshold) / self.theta_threshold_radians )
            target_reward = -(abs(target_position - current_position) ** 2) / ((2 * self.x_threshold) ** 2)
            return ( 1 + self.target_desire_factor * target_reward + (1 - self.target_desire_factor) * angle_reward )
        #TODO 2.2: Implementar 2 funcions de reward extra

        elif self.reward_function == "custom1":
            #Custom Reward 1: Recompensa basada en la distància entre la posició del carret i el target
            #Penalitza més l'angle i premia si la posició del carret és a prop del target
            angle_penalty = abs(angle) / self.theta_threshold_radians
            distance_to_target = abs(target_position - current_position)
            target_reward = 1 - (distance_to_target / self.x_threshold) ** 2
            return max(0, target_reward - angle_penalty)
        elif self.reward_function == "custom2":
            #Custom Reward 2: Recompensa per llindar
            #Penalització de l'angle
            angle_penalty = abs(angle) / self.theta_threshold_radians
            #Distància al target
            distance_to_target = abs(target_position - current_position)
            #Penalització contínua de la distància (inspirada en custom1)
            target_reward = 1 - (distance_to_target / self.x_threshold) ** 2
            #Llindar per recompensa extra
            distance_threshold = 0.2  # Llindar de 0.2 unitats al voltant del target
            if distance_to_target <= distance_threshold:
                target_reward += 0.5  # Bonificació si és prou a prop del target
            # Combinació de les components
            return max(0, target_reward - angle_penalty)

        else:
            raise AttributeError("Invalid reward function specified")

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot, _ = self.state
        if self.increased_actions:
            force_factor = action - 3
            force = (
                force_factor / 3 * self.force_mag
                if force_factor > 0
                else (force_factor - 1) / 3 * self.force_mag
            )
        else:
            force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array(
            (x, x_dot, theta, theta_dot, self.target_position), dtype=np.float64
        ) #Aquest estat es diferent respecte al del cartpole original

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = self.custom_reward(self.target_position, x, theta, terminated)

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        super().reset(seed=seed)
        self.steps = 0
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high

        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).tolist()
        self.steps_beyond_terminated = None

        #Generar posició aleatòria del target i guardar en l'estat
        self.target_position = self.generate_random_target_position()
        self.state.append(self.target_position)

        self.state = np.array(self.state)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        targetwidth = 10
        targetheight = 10

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        l, r, t, b = (
            -targetwidth / 2,
            targetwidth / 2,
            targetheight / 2,
            -targetheight / 2,
        )
        axleoffset = targetheight / 4.0
        targetx = x[-1] * scale + self.screen_width / 2.0  # MIDDLE OF target
        targety = 90  # TOP OF target
        target_coords = [(l, b), (l, t), (r, t), (r, b)]
        target_coords = [(c[0] + targetx, c[1] + targety) for c in target_coords]
        gfxdraw.aapolygon(self.surf, target_coords, (10, 255, 10))
        gfxdraw.filled_polygon(self.surf, target_coords, (10, 255, 10))

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
