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
        0	Mirror angle                20          70
        1   Goal location               0           100

    Actions:
        Type: Discrete(4)
        Num	Action
        0	Decrease
        1	Increase

    Reward:
        Reward is -1 for every step taken for every step NOT hitting the goal.
        Reward is 1 for every step < 5 from the goal.

    Starting State:
        The mirror starts randomly between 22.5 and 67.5.
        The goal starts randomly between 0 and 100

    Episode Termination:
        100 Steps have passed.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        self.screen_size = 700

        self.lowest_angle = 20
        self.highest_angle = 70

        self.lowest_goal = 0
        self.highest_goal = 100

        self.low = np.array([self.lowest_angle, self.lowest_goal])
        self.high = np.array([self.highest_angle, self.highest_goal])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.low, self.high)

        self.stepnr = 0

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state

        # Move the mirror
        if action == 0:
            if state[0] > self.lowest_angle:
                state[0] = state[0] - 1
        else:
            if state[0] < self.highest_angle:
                state[0] = state[0] + 1

        # Compute reward based on old goal location
        goal_location = (state[1] / 100) * self.screen_size
        real_location = ((self.highest_angle - state[0]) / (self.highest_angle - self.lowest_angle)) * self.screen_size

        reward = -1
        if math.pow(goal_location - real_location, 2) < math.pow(self.screen_size / 20, 2):
            reward = 1

        # Change goal location
        state[1] = state[1] + np.random.rand() * 4 - 2
        if state[1] < 0 or state[1] > 100:
            state[1] = self.state[1]

        self.stepnr = self.stepnr + 1

        done = False if self.stepnr < 200 else True

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.angle = self.np_random.rand() * (self.highest_angle - self.lowest_angle) + self.lowest_angle
        self.goal = self.np_random.rand() * (self.highest_goal - self.lowest_goal) + self.lowest_goal
        self.state = [self.angle, self.goal]
        self.stepnr = 0
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = self.screen_size
        screen_height = self.screen_size

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
            first_beam.set_color(1, 1, 0)
            self.viewer.add_geom(first_beam)

            # Second light beam
            self.second_beam = rendering.Line((screen_width / 2, screen_height / 2),
                                              (screen_width / 2, screen_height))
            self.second_beam.set_color(1, 1, 0)
            self.viewer.add_geom(self.second_beam)

            # Goal
            self.goal = rendering.make_circle(self.screen_size / 40)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.goal.set_color(0, .5, 0)
            self.goaltrans.set_translation(screen_width * (self.state[1] / 100), screen_height)
            self.viewer.add_geom(self.goal)

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

        self.mirrortrans.set_rotation(x[0] / 57.29577951308232)

        beam_pos = (self.highest_angle - x[0]) / (self.highest_angle - self.lowest_angle)
        self.second_beam.__init__((screen_width / 2, screen_height / 2), (beam_pos * screen_width, screen_height))
        self.second_beam.set_color(1, 1, 0)

        self.goaltrans.set_translation(screen_width * (x[1] / 100), screen_height)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def close(self):
    if self.viewer: self.viewer.close()
