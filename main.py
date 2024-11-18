#!/usr/bin/env python3

import os
import argparse
from typing import Iterator, Tuple, List

from interfaces.catanatron_interface import CatanatronParser
from environment.board_state import StaticBoardState
from environment.game import CatanGame, GameStep

# DEBUG LOGGING
VERBOSE_LOGGING = False

# INPUT TENSOR SIZE CHECKS
ENABLE_RUNTIME_TENSOR_SIZE_CHECKS = True
FLATTENED_STATIC_BOARD_STATE_LENGTH = 1817
EXPECTED_NUMBER_OF_PLAYERS = 4
FLATTENED_PLAYER_STATE_LENGTH = 26
FLATTENED_DYNAMIC_BOARD_STATE_LENGTH = 1254
FLATTENED_ACTION_LENGTH = 13
INPUT_TENSOR_EXPECTED_LENGTH = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH + FLATTENED_ACTION_LENGTH

def process_directory_iterator(base_dir: str) -> Iterator[Tuple[str, str]]:
    for root, dirs, _ in os.walk(base_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            board_file = os.path.join(subdir_path, "board.json")
            data_file = os.path.join(subdir_path, "data.json")

            if os.path.exists(board_file) and os.path.exists(data_file):
                yield board_file, data_file

def parse_data(board_path, data_path) -> Tuple[StaticBoardState, CatanGame]:
    parser = CatanatronParser()
    static_board_state = parser.parse_board_json(board_path)
    game = parser.parse_data_json(data_path, static_board_state)
    return [static_board_state, game]

def create_input_tensor(board_state: StaticBoardState, step: GameStep):
    input_tensor = []
    player_states, dynamic_board_state, action_taken = step.step
        
    input_tensor.extend(board_state.flatten())
    
    if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
        assert len(input_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH, "Static board state tensor unexpected size!"

    for player_state in player_states:
        input_tensor.extend(player_state.flatten())
    
    if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
        assert len(player_states) == EXPECTED_NUMBER_OF_PLAYERS, "Unexpected number of players!"
        assert len(input_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH, "Player state tensor unexpected size!"

    
    input_tensor.extend(dynamic_board_state.flatten())
    if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
        assert len(input_tensor) == FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH, "Dynamic board state tensor unexpected size!"

    input_tensor.extend(action_taken.flatten())
    if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
        assert len(input_tensor) == INPUT_TENSOR_EXPECTED_LENGTH, "Action tensor unexpected size!"

    return input_tensor

def main():

    parser = argparse.ArgumentParser(description="Parse Catan board.json and data.json files in subdirectories within a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the base directory containing training dataset.")
    args = parser.parse_args()

    # Process the directory
    if os.path.isdir(args.dataset_dir):
        for board_path, data_path in process_directory_iterator(args.dataset_dir):
            print(f"Processing: {board_path}, {data_path}")
            static_board_state, game = parse_data(board_path, data_path)

            for step in game.game_steps:
                input_tensor = create_input_tensor(static_board_state, step)

                if VERBOSE_LOGGING:
                    print(f"Input Tensor Size: {len(input_tensor)}")
                    print(f"Input Tensor:\n {input_tensor}")
                
                if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
                    assert all(isinstance(value, int) for value in input_tensor), "Input tensor contains non-integer values!"
                    assert len(input_tensor) == INPUT_TENSOR_EXPECTED_LENGTH, "Input tensor unexpected size!"
    else:
        print(f"The specified path {args.dataset_dir} is not a directory.")


if __name__ == "__main__":
    main()
