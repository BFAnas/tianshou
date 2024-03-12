import gymnasium as gym
import pygame
import math
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from pygame.surfarray import array3d

class PointMass(gym.Env):
    def __init__(self, risk_penalty, high_state=1, low_state=0, risk_prob=0.7, max_steps=200, goal=np.array([0.1, 0.1])):
        # Step 1: Car parameters
        self.v_max = 0.05
        self.v_sigma = 0.01
        # Step 3: Environment parameters
        self.d_goal = 0.01
        self.d_sampling = 0.1
        self.init_pos = np.array([1.0, 1.0])
        self.risk_prob = risk_prob
        self.risk_penalty = risk_penalty

        self.low_state = low_state
        self.high_state= high_state

        self.min_actions = np.array(
            [-self.v_max, -self.v_max], dtype=np.float32
        )
        self.max_actions = np.array(
            [self.v_max, self.v_max], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(2, ),
            dtype=np.float32
        )

        self.goal = goal
        self.r = 0.3 # obstacle radius
        self.center = np.array([0.5, 0.5])

        # Step 4: Rendering parameters
        self.screen_size = [600, 600]
        self.screen_scale = 600
        self.background_color = [255, 255, 255]
        self.wall_color = [0, 0, 0]
        self.circle_color = [255, 0, 0]
        self.safe_circle_color = [200,0,0]
        self.lidar_color = [0, 0, 255]
        self.goal_color = [0, 255, 0]
        self.robot_color = [0, 0, 0]
        self.safety_color = [255, 0, 0]
        self.goal_size = 15
        self.radius = 9
        self.width = 3
        self.pygame_init = False

        # Initialize the trajectory list
        self.trajectory = []

        # Maximum number of steps per episode
        self.max_steps = max_steps
        self.current_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        return [seed]

    def reset(self, eval=False, init_state=None):
        sampled = False
        if init_state is not None:
            self.state = init_state
        else:
            while not sampled:
                # uniform state space initial state distribution
                self.state = self.observation_space.sample()

                if self.is_safe():
                    sampled = True
            if eval:
                self.state = np.array([0.85, 0.85]) + self.np_random.uniform(-0.05, 0, size=(2,))

        # Reset the trajectory list
        self.trajectory = [self.state.copy()]
        self.current_step = 0 # Reset the step count

        return np.array(self.state), {}


    def get_dist_to_goal(self):
        return np.linalg.norm(self.state-self.goal)

    # Check if the state is safe.
    def is_safe(self):
        return np.linalg.norm(self.state-self.center) > self.r

    def get_reward(self):
        d_goal = self.get_dist_to_goal()
        reward = -d_goal - .1
        cost = 0
        if not self.is_safe():
            u = np.random.uniform(0, 1)
            if u > self.risk_prob:
                cost = 1
                reward -= self.risk_penalty
        return reward, cost

    def step(self, action):
        self.current_step += 1
        self.state = self.state + action
        
        terminated = 0
        if self.get_dist_to_goal() < self.d_goal:
            terminated = 1

        reward, cost = self.get_reward()
        
        # Truncate the episode after max_steps
        truncated = 0
        if (self.current_step >= self.max_steps):
            truncated = 1

        if terminated:
            reward += 1

        # Update the trajectory with the new position
        self.trajectory.append(self.state.copy())

        return np.array(self.state), reward, terminated, truncated, {'cost':cost}

    def render(self):
        if not self.pygame_init:
            pygame.init()
            self.pygame_init = True
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(self.background_color)
        p_car = self.state
        p = (self.screen_scale * p_car).astype(int).tolist()

        c, r = (self.screen_scale*self.center).astype(int), int(self.screen_scale*self.r)
        pygame.draw.circle(self.screen, self.circle_color, c, r)
        pygame.draw.circle(self.screen, self.goal_color, (self.screen_scale * self.goal).astype(int), self.goal_size)
        pygame.draw.circle(self.screen, self.robot_color, p, self.radius, self.width)

        # Draw the trajectory
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                start_pos = (self.screen_scale * self.trajectory[i]).astype(int)
                end_pos = (self.screen_scale * self.trajectory[i + 1]).astype(int)
                pygame.draw.line(self.screen, self.lidar_color, start_pos, end_pos, 2)

        self.clock.tick(20)
        # Convert the Pygame surface to a numpy array
        frame = array3d(pygame.display.get_surface())

        return frame