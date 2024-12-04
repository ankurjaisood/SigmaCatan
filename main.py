#!/usr/bin/env python3

import os
import argparse
import random
import numpy as np
from typing import Generator, Tuple, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from interfaces.catanatron_interface import CatanatronParser
from environment.board_state import StaticBoardState, DynamicBoardState
from environment.player_state import PlayerState
from environment.game import CatanGame
from environment.action import Action, ActionType
from rewards.reward_functions import BasicRewardFunction
from agents.dqn import DQNTrainer

import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEBUG LOGGING
VERBOSE_LOGGING = False

# INPUT TENSOR SIZE CHECKS
ENABLE_RUNTIME_TENSOR_SIZE_CHECKS = True
FLATTENED_STATIC_BOARD_STATE_LENGTH = 1817
EXPECTED_NUMBER_OF_PLAYERS = 4
FLATTENED_PLAYER_STATE_LENGTH = 26
FLATTENED_DYNAMIC_BOARD_STATE_LENGTH = 1254
INPUT_STATE_TENSOR_EXPECTED_LENGTH = 2407
OUTPUT_TENSOR_EXPECTED_LENGTH = (len(ActionType) + 1) * 1000
FLATTENED_ACTION_LENGTH = 1

class GameIterator:
    def __init__(self, dir_path: str) -> None:
        self.parser = CatanatronParser()
        self.games_paths = np.array(self.process_directory_iterator(dir_path), dtype=object)
        self.games = self.get_games()
        random.shuffle(self.games)

    def process_directory_iterator(self, base_dir: str) -> List[Tuple[str, str]]:
        base_path = Path(base_dir)
        return [
            (str(board_file), str(data_file))
            for subdir in base_path.iterdir() if subdir.is_dir()
            for board_file in [subdir / "board.json"]
            for data_file in [subdir / "data.json"]
            if board_file.exists() and data_file.exists()
        ]

    def parse_data(self, board_path, data_path) -> Tuple[StaticBoardState, CatanGame]:
        static_board_state = self.parser.parse_board_json(board_path)
        game = self.parser.parse_data_json(data_path, static_board_state)
        return static_board_state, game

    def get_games(self) -> List[Tuple[StaticBoardState, CatanGame]]:
        with ThreadPoolExecutor() as executor:
            games = list(executor.map(lambda paths: self.parse_data(*paths), self.games_paths))
        return games

    def create_action_tensor(self, action_taken: Action):
        input_action_tensor = np.array(action_taken.flatten(), dtype=np.float32)
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert input_action_tensor.size == FLATTENED_ACTION_LENGTH, "Action tensor unexpected size!"
        return int(input_action_tensor[0]) if input_action_tensor.size == 1 else input_action_tensor

    def create_state_tensor(self, board_state: StaticBoardState, dynamic_board_state: DynamicBoardState, player_states: List[PlayerState]):
        static_state = np.array(board_state.flatten(), dtype=np.float32)
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert static_state.size == FLATTENED_STATIC_BOARD_STATE_LENGTH, "Static board state tensor unexpected size!"

        player_states_array = np.array([ps.flatten() for ps in player_states], dtype=np.float32).flatten()
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(player_states) == EXPECTED_NUMBER_OF_PLAYERS, "Unexpected number of players!"
            assert player_states_array.size == EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH, "Player state tensor unexpected size!"

        dynamic_state = np.array(dynamic_board_state.flatten(), dtype=np.float32)
        state_tensor = np.concatenate((static_state, player_states_array, dynamic_state))
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert state_tensor.size == INPUT_STATE_TENSOR_EXPECTED_LENGTH, f"Dynamic board state tensor unexpected size! {state_tensor.size}"

        return state_tensor

    def iterate_game(self) -> Generator[list, None, None]:
        for game_index, game_data in enumerate(self.games):
            static_board_state, game = game_data
            reward_function = BasicRewardFunction(game.winner)
            for index, step in enumerate(game.game_steps):
                player_states, dynamic_board_state, action_taken = step.step

                if game.winner != dynamic_board_state.current_player:
                    if VERBOSE_LOGGING:
                        logger.debug(f"Skipping step {index} because it is not the winning player's turn.")
                    continue

                reward = reward_function.calculate_reward(step.get_player_state_by_ID(game.winner), action_taken)
                input_state_tensor = self.create_state_tensor(static_board_state, dynamic_board_state, player_states)
                input_action_tensor = self.create_action_tensor(action_taken)

                next_state = None
                try:
                    next_player_states, next_dynamic_board_state, _ = game.game_steps[index + 1].step
                    next_state = self.create_state_tensor(static_board_state, next_dynamic_board_state, next_player_states)
                except IndexError:
                    logger.info(f"Game: {game_index} No next state found! Game is over.")

                next_state_tensor = next_state if next_state is not None else np.full(INPUT_STATE_TENSOR_EXPECTED_LENGTH, -1, dtype=np.float32)
                game_finished_tensor = np.array([0] if next_state is not None else [1], dtype=np.float32)

                yield input_state_tensor, input_action_tensor, [reward], next_state_tensor, game_finished_tensor

    def __iter__(self):
        return self.iterate_game()

def main():
    parser = argparse.ArgumentParser(description="Parse Catan board.json and data.json files in subdirectories within a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the base directory containing training dataset.")
    args = parser.parse_args()

    if os.path.isdir(args.dataset_dir):
        input_tensor_expected_length = INPUT_STATE_TENSOR_EXPECTED_LENGTH
        output_tensor_expected_length = OUTPUT_TENSOR_EXPECTED_LENGTH
        dqn_trainer = DQNTrainer(input_tensor_expected_length, output_tensor_expected_length)
        game_iterator = GameIterator(args.dataset_dir)
        dqn_trainer.train(game_iterator)
    else:
        logger.error(f"The specified path {args.dataset_dir} is not a directory.")

if __name__ == "__main__":
    main()
