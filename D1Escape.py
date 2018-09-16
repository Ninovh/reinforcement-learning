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


class D1Escape(gym.Env):
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
        Type: Discrete(2)
        Num	Action
        0	Subtract 1
        1	Add 1

    Reward:
        Reward is -1 for every step taken, including the termination step

    Starting State:
        The player starts somewhere between 2 and 8.

    Episode Termination:
        Player is at location 0.
        Player is at location 10.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        self.low = np.array([0])
        self.high = np.array([10])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.low, self.high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state

        S1 = state
        S2 = state - 1
        S3 = state + 1

        if action == 0:
            self.state = state - 1
        else:
            self.state = state + 1

        reward = - 1
        done = False
        if self.state == 0 or self.state == 10:
            reward = 1
            done = True

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.randint(low=2, high=9, size=(1,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        playersize = 50

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
