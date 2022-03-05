#!/usr/bin/env python

# from utils import resize_image, XboxController, KeyboardController
from utils import resize_image
from bc_agent_model import MarioKartBCAgent
from termcolor import cprint

import gym
import gym_mupen64plus
import numpy as np
import argparse
import torch
# Play
class Actor(object):

    def __init__(self, controller_type, agent_path):
        # Load in model from train.py and load in the trained weights
        self.model = MarioKartBCAgent()
        self.model.load_state_dict(torch.load(agent_path))

        # Init controller
        # if controller_type == "keyboard":
        #     self.controller = KeyboardController()
        # elif controller_type == "xbox":
        #     self.controller = XboxController()


    def get_action(self, obs):

        ### determine manual override
        # manual_override = self.real_controller.LeftBumper == 1
        manual_override = False
        if not manual_override:
            vec = resize_image(obs)
            vec = np.expand_dims(vec, axis=0) # expand dimensions for predict, it wants (1,66,200,3) not (66, 200, 3)
            vec = np.transpose(vec, (0, 3, 1, 2))
            joystick = self.model(torch.tensor(vec).float()).detach().numpy()[0]

        else:
            joystick = self.real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)


        ## Act

        ### calibration
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]
        ### print to console
        if manual_override:
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter Controller type.')
    parser.add_argument('-controller', '-c', type=str, default='keyboard', help=
            'Options: xbox, keyboard.  Default: keyboard'
        )
    parser.add_argument('-agent_path', '-a', type=str, default='', help=
            'Enter path of your trained agent weights'
        )
    parser.add_argument('-track', '-t', type=str, default='Mario-Kart-Luigi-Raceway-v0', help=
         'Enter the racetrack name you want to evaluate on'
    )
    args = parser.parse_args()
    env = gym.make(args.track)

    obs = env.reset()
    print('env ready!')

    actor = Actor(args.controller, args.agent_path)
    print('actor ready!')

    print('beginning episode loop')
    total_reward = 0
    end_episode = False
    while not end_episode:
        action = actor.get_action(obs)
        obs, reward, end_episode, info = env.step(action)
        total_reward += reward

    print('end episode... total reward: ' + str(total_reward))

    obs = env.reset()
    print('env ready!')

    input('press <ENTER> to quit')

    env.close()
