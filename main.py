#!/usr/bin/env python3

import os
import argparse
from typing import Generator, Iterator, Tuple, List

from interfaces.catanatron_interface import CatanatronParser
from environment.board_state import StaticBoardState, DynamicBoardState
from environment.player_state import PlayerState
from environment.game import CatanGame, GameStep
from environment.action import Action, ActionType
from rewards.reward_functions import VPRewardFunction, BasicRewardFunction
from agents.dqn import DQNTrainer

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

    def create_action_tensor(self, 
                            action_taken: Action):
        input_action_tensor = []
        input_action_tensor.extend(action_taken.flatten())
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(input_action_tensor) == FLATTENED_ACTION_LENGTH, "Action tensor unexpected size!"
        return input_action_tensor

    def create_state_tensor(self, 
                            board_state: StaticBoardState, 
                            dynamic_board_state: DynamicBoardState,
                            player_states: List[PlayerState]):
        state_tensor = []

        state_tensor.extend(board_state.flatten())
        
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(state_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH, "Static board state tensor unexpected size!"

        for player_state in player_states:
            state_tensor.extend(player_state.flatten())
        
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(player_states) == EXPECTED_NUMBER_OF_PLAYERS, "Unexpected number of players!"
            assert len(state_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH, "Player state tensor unexpected size!"

        state_tensor.extend(dynamic_board_state.flatten())
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(state_tensor) == INPUT_STATE_TENSOR_EXPECTED_LENGTH, "Dynamic board state tensor unexpected size!"

        return state_tensor
    
    def iterate_game(self) -> Generator[list, list, float]:
        for index, step in enumerate(self.game.game_steps):
            player_states, dynamic_board_state, action_taken = step.step

            # skip this turn if its not the winning players turn (we are the winning player)
            if self.game.winner != dynamic_board_state.current_player:
                if VERBOSE_LOGGING: print(f"Skipping step {index} because it is not the winning player's turn.")
                continue

            reward = self.reward_function.calculate_reward(step.get_player_state_by_ID(self.game.winner), action_taken)
            input_state_tensor = self.create_state_tensor(
                self.static_board_state, 
                dynamic_board_state, 
                player_states)
            input_action_tensor = self.create_action_tensor(action_taken)
            
            # TODO(jaisood): Is there a better way to do this
            next_state = None
            try:
                next_player_states, next_dynamic_board_sate, _ = self.game.game_steps[index + 1].step
                next_state = self.create_state_tensor(
                    self.static_board_state,
                    next_dynamic_board_sate,
                    next_player_states)
            except IndexError as e:
                print(f"No next state prime found! Game is over!")

            next_state_tensor = next_state if next_state is not None else [-1] * INPUT_STATE_TENSOR_EXPECTED_LENGTH
            game_finished_tensor = [0] if next_state is not None else [1]

            yield input_state_tensor, input_action_tensor, [reward], next_state_tensor, game_finished_tensor

    def __iter__(self):
        # Make the class iterable by returning the generator
        return self.iterate_game()
            
def main():

    parser = argparse.ArgumentParser(description="Parse Catan board.json and data.json files in subdirectories within a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the base directory containing training dataset.")
    args = parser.parse_args()

    # Process the directory
    if os.path.isdir(args.dataset_dir):
        input_tensor_expected_length = INPUT_STATE_TENSOR_EXPECTED_LENGTH + FLATTENED_ACTION_LENGTH
        output_tensor_expected_length = max(ActionType)
        dqn_trainer = DQNTrainer(input_tensor_expected_length, output_tensor_expected_length)

        for board_path, data_path in process_directory_iterator(args.dataset_dir):
            game_iterator = GameIterator(board_path, data_path)
            dqn_trainer.train(game_iterator)

    else:
        print(f"The specified path {args.dataset_dir} is not a directory.")


if __name__ == "__main__":
    main()
