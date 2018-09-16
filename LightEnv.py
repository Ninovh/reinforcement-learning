"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering


class D2Escape(gym.Env):
    """
    Description:
        The player starts somewhere between 2 and 8 and needs to get to either 0 or 10 by adding or subtracting 1.
    Source:
        Test
    Observation:
        Type: Box(1)
        Num	Observation                 Min         Max
        0	Player Location             0           10

    Actions:
        Type: Discrete(4)
        Num	Action
        0	Up
        1	Left
        2   Down
        3   Right

    Reward:
        Reward is -1 for every step taken.

    Starting State:
        The player starts somewhere between (3,3) and (7,7).

    Episode Termination:
        Player is at location (0,0).
        Player is at location (10,10).
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, size, error_chance):

        self.error_chance = error_chance

        self.lowest = 0
        self.highest = size - 1

        self.low = np.array([self.lowest, self.lowest])
        self.high = np.array([self.highest, self.highest])

        self.traps = [self.lowest + int((self.highest - self.lowest) / 4),
                      self.highest - int((self.highest - self.lowest) / 4)]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.low, self.high)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state

        # Simulate errors
        if self.error_chance > 0:
            action = self.determine_action(action)

        # Move player according to action
        self.move_player(action, state)

        # Reset to valid position
        reward = -99
        if self.state[0] < self.lowest:
            self.state[0] = self.lowest
        elif self.state[1] < self.lowest:
            self.state[1] = self.lowest
        elif self.state[0] > self.highest:
            self.state[0] = self.highest
        elif self.state[1] > self.highest:
            self.state[1] = self.highest
        # If position is valid already
        else:
            reward = -1

        # If on a trap
        for i in self.traps:
            for j in self.traps:
                if self.state[0] == i and self.state[1] == j:
                    reward = -99

        done = False
        # If on exit, game finished
        for i in [self.lowest, self.highest]:
            for j in [self.lowest, self.highest]:
                if self.state[0] == i and self.state[1] == j:
                    reward = 99
                    done = True

        return np.array(self.state), reward, done, {}

    def move_player(self, action, state):
        if action == 0:
            self.state = state + [0, -1]
        elif action == 1:
            self.state = state + [-1, 0]
        elif action == 2:
            self.state = state + [0, 1]
        else:  # action == 3
            self.state = state + [1, 0]

    def determine_action(self, action):
        chance = np.random.rand()
        if chance > 1 - (self.error_chance / 3):
            action = (action + 1) % 4
            pass
        elif chance > 1 - 2 * (self.error_chance / 3):
            action = (action + 2) % 4
            pass
        elif chance > 1 - 3 * (self.error_chance / 3):
            action = (action + 3) % 4
            pass
        else:
            pass
        return action

    def reset(self):
        self.state = self.np_random.randint(low=self.lowest + (self.highest - self.lowest) * .4,
                                            high=self.highest - (self.highest - self.lowest) * .4 + 2, size=(2,))
        return np.array(self.state)

    def render(self, mode='human'):
        screen_size = 700
        screen_width = screen_size
        screen_height = screen_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Background
            # background = rendering.FilledPolygon([(0, 0), (0, screen_height), (screen_width, screen_height), (screen_width, 0)])
            # background.set_color(1, 1, 1)
            # self.viewer.add_geom(background)

            # Start of the light beam
            start = rendering.FilledPolygon(
                [(0 + screen_width / 30, screen_height / 2 + screen_height / 20),
                 (0, screen_height / 2 + screen_height / 20),
                 (0, screen_height / 2 - screen_height / 20),
                 (0 + screen_width / 30, screen_height / 2 - screen_height / 20)])
            self.viewer.add_geom(start)

            # First light beam
            first_beam = rendering.Line((0 + screen_width / 30, screen_height / 2),
                                        (screen_width / 2, screen_height / 2))
            first_beam.set_color(255, 255, 0)
            self.viewer.add_geom(first_beam)

            # Second light beam
            self.second_beam = rendering.Line((screen_width / 2, screen_height / 2),
                                              (screen_width / 2, screen_height))
            self.second_beam.set_color(255, 255, 0)
            self.viewer.add_geom(self.second_beam)

            # Goal
            goal = rendering.make_circle(screen_size / 40)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goal.set_color(0, .5, 0)
            self.goaltrans.set_translation(screen_width / 2, screen_height)
            self.viewer.add_geom(goal)

            # Mirror
            mirror_w = screen_width / 30
            mirror_h = screen_height / 500
            mirror = rendering.FilledPolygon(
                [(- mirror_w, - mirror_h), (- mirror_w, + mirror_h), (mirror_w, + mirror_h), (mirror_w, - mirror_h)])
            mirror.set_color(.8, .6, .4)
            self.mirrortrans = rendering.Transform()
            self.mirrortrans.set_translation(screen_width / 2, screen_height / 2)
            mirror.add_attr(self.mirrortrans)
            self.viewer.add_geom(mirror)

            if self.state is None:
                return None

        x = self.state

        # self.mirrortrans.set_rotation(self.mirrortrans.rotation + .01)

        self.mirrortrans.set_rotation(67.5/57.29577951308232)
        self.second_beam = rendering.make_polyline([(0,0),(500,500)])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
