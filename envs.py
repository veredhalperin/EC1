from collections import deque
from logging import ERROR
import atari_py
import cv2
import gym
from gym import spaces
import numpy as np
import torch
import random
import pandas as pd

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger


class CPCEnv():
    def __init__(self, args):
        self.device = args.device
        self.actions = dict([i, e] for i, e in zip(range(2), [0, 1]))
        self.training = True  # Consistent with model training mode
        self.Trial = 6
        self.done = False
        df = pd.read_csv('final.csv')
        df=df[df['SubjID']==10100]
        self.all_data = df.drop(columns=['Apay', 'Bpay', 'Order'])
        self.rewards = df[['Apay', 'Bpay', 'SubjID', 'GameID', 'Trial']]
        self.IDs = self.all_data['SubjID'].unique()
        random.shuffle(self.IDs)
        self.num_player = 0
        self.GameIDs = \
        df[df['SubjID'] == self.IDs[self.num_player]][['GameID', 'Order']].drop_duplicates().sort_values(by=['Order'])[
            'GameID'].values
        self.num_game = 0
        self.window = 58
        self.state_buffer = deque([], maxlen=self.window)
        self.ale = atari_py.ALEInterface()
        self.ale.loadROM(atari_py.get_game_path(
            args.game))  # ROM loading must be done after setting options,Loads & initializes a game
        self.decisions = {}

    def num_columns(self):
        return len(self.all_data.drop(columns=['Trial','GameID','SubjID']).columns)

    @property
    def action_space(self):
        return spaces.Discrete(len(self.actions))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, dtype=np.float32, shape=(self.window, 1, self.num_columns()))

    def _get_state(self):
        return torch.tensor(np.array(self.all_data[(self.all_data['Trial'] == self.Trial) & (
                self.all_data['GameID'] == self.GameIDs[self.num_game]) & (self.all_data['SubjID'] == self.IDs[
            self.num_player])].drop(columns=['Trial','GameID','SubjID'])), dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(1, self.num_columns(), device=self.device))

    def reset(self):
        # Reset internals
        self._reset_buffer()
        # Process & return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)

    def act(self, action):
        self.decisions[(self.IDs[self.num_player], self.GameIDs[self.num_game], self.Trial)] = action
        if self.Trial != 6:
            self.state_buffer.pop()
        if self.Trial < 25:
            self.Trial += 1
        elif self.num_game < len(self.GameIDs) - 1:
            self.Trial = 6
            self.num_game += 1
        elif self.num_player < len(self.IDs) - 1:
            self.Trial = 6
            self.num_game = 0
            self.num_player += 1
            self.GameIDs = self.all_data[self.all_data['SubjID'] == self.IDs[self.num_player]]['GameID'].unique()
            self.reset()
        else:
            self.done = True
        if action == 1:
            return self.rewards[
                (self.rewards['Trial'] == self.Trial) & (self.rewards['GameID'] == self.GameIDs[self.num_game]) & (
                        self.rewards['SubjID'] == self.IDs[self.num_player])]['Bpay'].values[0]
        else:
            return self.rewards[
                (self.rewards['Trial'] == self.Trial) & (self.rewards['GameID'] == self.GameIDs[self.num_game]) & (
                        self.rewards['SubjID'] == self.IDs[self.num_player])]['Apay'].values[0]

    def game_over(self):
        return self.done

    def step(self, action):
        reward = self.act(self.actions.get(action))
        self.state_buffer.append(self._get_state())
        return torch.stack(list(self.state_buffer), 0), reward, self.game_over()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :,
                             ::-1])  # imshow displays an image in a window, getScreenRGB fills screen_data with the data in RGB format
        cv2.waitKey(1)  # eyboard binding function

    def close(self):
        cv2.destroyAllWindows()  # simply destroys all the windows we created

    @property
    def hash_space(self):
        return spaces.Box(low=0, high=1, dtype=np.float32, shape=(128,))

    @property
    def get_state_hash(self):
        return self.ale.getRAM().astype(np.float32) / 255  # Returns the current RAM content


class AtariEnv():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)  # Set the value of a setting
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions,Set the value of a setting
        self.ale.setInt('frame_skip', 0)  # Set the value of a setting
        self.ale.setBool('color_averaging', False)  # Set the value of a setting
        self.ale.loadROM(atari_py.get_game_path(
            args.game))  # ROM loading must be done after setting options,Loads & initializes a game
        actions = self.ale.getMinimalActionSet()  # Returns the vector of the minimal set of actions needed to playthe game
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    @property
    def action_space(self):
        return spaces.Discrete(len(self.actions))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, dtype=np.float32, shape=(4, 84, 84))

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84),
                           interpolation=cv2.INTER_LINEAR)  # this function fills screen_data with the data in grayscale
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life, Applies an action to the game & returns the reward
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()  # Resets the game, but not the full system
            # Perform up to 30 random no-ops before starting
            for _ in range(np.random.randint(30)):
                self.ale.act(
                    0)  # Assumes raw action 0 is always no-op, Applies an action to the game & returns the reward
                if self.ale.game_over():  # Indicates if the game has ended
                    self.ale.reset_game()  # Resets the game, but not the full system
        # Process & return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()  # The remaining number of lives
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))  # Applies an action to the game & returns the reward
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()  # Indicates if the game has ended
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()  # The remaining number of lives
            if lives < self.lives & lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :,
                             ::-1])  # imshow displays an image in a window, getScreenRGB fills screen_data with the data in RGB format
        cv2.waitKey(1)  # eyboard binding function

    def close(self):
        cv2.destroyAllWindows()  # simply destroys all the windows we created

    @property
    def hash_space(self):
        return spaces.Box(low=0, high=1, dtype=np.float32, shape=(128,))

    @property
    def get_state_hash(self):
        return self.ale.getRAM().astype(np.float32) / 255  # Returns the current RAM content
