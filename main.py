#!/usr/bin/env python3

import os
import argparse
from typing import Generator, Iterator, Tuple, List

from interfaces.catanatron_interface import CatanatronParser
from environment.board_state import StaticBoardState, DynamicBoardState
from environment.player_state import PlayerState
from environment.game import CatanGame, GameStep
from environment.action import Action
from rewards.reward_functions import VPRewardFunction, BasicRewardFunction

# DEBUG LOGGING
VERBOSE_LOGGING = False

# INPUT TENSOR SIZE CHECKS
ENABLE_RUNTIME_TENSOR_SIZE_CHECKS = True
FLATTENED_STATIC_BOARD_STATE_LENGTH = 1817
EXPECTED_NUMBER_OF_PLAYERS = 4
FLATTENED_PLAYER_STATE_LENGTH = 26
FLATTENED_DYNAMIC_BOARD_STATE_LENGTH = 1254
INPUT_STATE_TENSOR_EXPECTED_LENGTH = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH

FLATTENED_ACTION_LENGTH = 13

def process_directory_iterator(base_dir: str) -> Iterator[Tuple[str, str]]:
    for root, dirs, _ in os.walk(base_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            board_file = os.path.join(subdir_path, "board.json")
            data_file = os.path.join(subdir_path, "data.json")

            if os.path.exists(board_file) and os.path.exists(data_file):
                yield board_file, data_file

class GameIterator:
    def __init__(self,
                 board_path: str,
                 data_path: str) -> None:
        
        print(f"Processing: {board_path}, {data_path}")
        self.static_board_state, self.game = self.parse_data(board_path, data_path)
        self.reward_function = BasicRewardFunction(self.game.winner)

    def parse_data(self, board_path, data_path) -> Tuple[StaticBoardState, CatanGame]:
        parser = CatanatronParser()
        static_board_state = parser.parse_board_json(board_path)
        game = parser.parse_data_json(data_path, static_board_state)
        return [static_board_state, game]

    def create_input_tensor(self, 
                            board_state: StaticBoardState, 
                            dynamic_board_state: DynamicBoardState,
                            player_states: List[PlayerState],
                            action_taken: Action):
        input_state_tensor = []
        input_action_tensor = []

        input_state_tensor.extend(board_state.flatten())
        
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(input_state_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH, "Static board state tensor unexpected size!"

        for player_state in player_states:
            input_state_tensor.extend(player_state.flatten())
        
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(player_states) == EXPECTED_NUMBER_OF_PLAYERS, "Unexpected number of players!"
            assert len(input_state_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH, "Player state tensor unexpected size!"

        input_state_tensor.extend(dynamic_board_state.flatten())
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(input_state_tensor) == INPUT_STATE_TENSOR_EXPECTED_LENGTH, "Dynamic board state tensor unexpected size!"

        input_action_tensor.extend(action_taken.flatten())
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(input_action_tensor) == FLATTENED_ACTION_LENGTH, "Action tensor unexpected size!"

        return input_state_tensor, input_action_tensor
    
    def iterate_game(self) -> Generator[list, list, float]:
        for step in self.game.game_steps:
            player_states, dynamic_board_state, action_taken = step.step
            reward = self.reward_function.calculate_reward(step.get_player_state_by_ID(self.game.winner))
            input_state_tensor, input_action_tensor = self.create_input_tensor(
                self.static_board_state, 
                dynamic_board_state, 
                player_states, 
                action_taken)
            
            yield input_state_tensor, input_action_tensor, reward

    def __iter__(self):
        # Make the class iterable by returning the generator
        return self.iterate_game()
            
def main():

    parser = argparse.ArgumentParser(description="Parse Catan board.json and data.json files in subdirectories within a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the base directory containing training dataset.")
    args = parser.parse_args()

    # Process the directory
    if os.path.isdir(args.dataset_dir):
        for board_path, data_path in process_directory_iterator(args.dataset_dir):
            print(f"Processing: {board_path}, {data_path}")
            game_iterator = GameIterator(board_path, data_path)

            for input_state_tensor, input_action_tensor, reward in game_iterator:
                if VERBOSE_LOGGING:
                    print(f"Input State Tensor Size: {len(input_state_tensor)}")
                    print(f"Input State Tensor:\n {input_state_tensor}")
                    print(f"Input Action Tensor Size: {len(input_action_tensor)}")
                    print(f"Input Action Tensor:\n {input_action_tensor}")
                    print(f"Reward: {reward}")

    else:
        print(f"The specified path {args.dataset_dir} is not a directory.")


if __name__ == "__main__":
    main()
