#!/usr/bin/env python3

import os
import argparse
import random
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
FLATTENED_DYNAMIC_BOARD_STATE_LENGTH = 486

INPUT_STATE_TENSOR_EXPECTED_LENGTH = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH
INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH
OUTPUT_TENSOR_EXPECTED_LENGTH = 14

FLATTENED_ACTION_LENGTH = 1

class GameIterator:
    def __init__(self,
                 dir_path: str,
                 static_board: bool) -> None:
        self.static_board = static_board
        self.parser = CatanatronParser()
        self.games_paths = self.process_directory_iterator(dir_path)
        self.games = self.get_games()
        random.shuffle(self.games) # Shuffle the games to prevent overfitting

    def process_directory_iterator(self, base_dir: str) -> List[Tuple[str, str]]:
        game_paths = []
        for root, dirs, _ in os.walk(base_dir):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                board_file = os.path.join(subdir_path, "board.json")
                data_file = os.path.join(subdir_path, "data.json")

                if os.path.exists(board_file) and os.path.exists(data_file):
                    game_paths.append((board_file, data_file))
        return game_paths
    
    def parse_data(self, board_path, data_path) -> Tuple[StaticBoardState, CatanGame]:
        static_board_state = self.parser.parse_board_json(board_path)
        game = self.parser.parse_data_json(data_path, static_board_state)
        return (static_board_state, game)
    
    def get_games(self) -> List[Tuple[StaticBoardState, CatanGame]]:
        games = []
        for board_path, data_path in self.games_paths:
            print(f"Processing: {board_path}, {data_path}")
            static_board_state, game = self.parse_data(board_path, data_path)
            games.append((static_board_state, game))
        return games

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

        if not self.static_board:
            state_tensor.extend(board_state.flatten())
            if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
                assert len(state_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH, "Static board state tensor unexpected size!"

        for player_state in player_states:
            state_tensor.extend(player_state.flatten())

        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert len(player_states) == EXPECTED_NUMBER_OF_PLAYERS, "Unexpected number of players!"
            if not self.static_board:
                assert len(state_tensor) == FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH, "Player state tensor unexpected size!"
            else:
                assert len(state_tensor) == EXPECTED_NUMBER_OF_PLAYERS*FLATTENED_PLAYER_STATE_LENGTH, "Static Board: Player state tensor unexpected size!"
        
        state_tensor.extend(dynamic_board_state.flatten())
        
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            if not self.static_board:
                assert len(state_tensor) == INPUT_STATE_TENSOR_EXPECTED_LENGTH, f"State tensor unexpected size! {len(state_tensor)}"
            else:
                assert len(state_tensor) == INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD, f"Static Board: State tensor unexpected size! {len(state_tensor)}"
        return state_tensor

    def iterate_game(self) -> Generator[list, list, float]:
        static_board_check_state = None
        for game_index, game_data in enumerate(self.games):
            static_board_state, game = game_data
            reward_function = BasicRewardFunction(game.winner)

            if self.static_board:
                if static_board_check_state is None:
                    static_board_check_state = static_board_state
                else:
                    assert static_board_check_state.flatten() == static_board_state.flatten(), f"Static board state is enabled but board state is not constant between games!"
            
            for index, step in enumerate(game.game_steps):
                player_states, dynamic_board_state, action_taken = step.step

                # skip this turn if its not the winning players turn (we are the winning player)
                if game.winner != dynamic_board_state.current_player:
                    if VERBOSE_LOGGING: print(f"Skipping step {index} because it is not the winning player's turn.")
                    continue

                reward = reward_function.calculate_reward(step.get_player_state_by_ID(game.winner), action_taken)
                input_state_tensor = self.create_state_tensor(
                    static_board_state,
                    dynamic_board_state,
                    player_states)
                input_action_tensor = self.create_action_tensor(action_taken)

                # TODO(jaisood): Is there a better way to do this
                next_state = None
                try:
                    next_player_states, next_dynamic_board_sate, _ = game.game_steps[index + 1].step
                    next_state = self.create_state_tensor(
                        static_board_state,
                        next_dynamic_board_sate,
                        next_player_states)
                except IndexError as e:
                    print(f"Game: {game_index} No next state prime found! Game is over!")

                next_state_tensor = next_state if next_state is not None else [-1] * (INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD if self.static_board else INPUT_STATE_TENSOR_EXPECTED_LENGTH)
                game_finished_tensor = [0] if next_state is not None else [1]

                yield input_state_tensor, input_action_tensor, [reward], next_state_tensor, game_finished_tensor

    def __iter__(self):
        # Make the class iterable by returning the generator
        return self.iterate_game()

def main():

    parser = argparse.ArgumentParser(description="Parse Catan board.json and data.json files in subdirectories within a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the base directory containing training dataset.")
    parser.add_argument("--static_board", action="store_true", help="Whether to expect a static board for all the games.")

    args = parser.parse_args()

    # Process the directory
    if os.path.isdir(args.dataset_dir):        
        if args.static_board:
            input_tensor_expected_length = INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD

        else:
            input_tensor_expected_length = INPUT_STATE_TENSOR_EXPECTED_LENGTH

        output_tensor_expected_length = OUTPUT_TENSOR_EXPECTED_LENGTH
        dqn_trainer = DQNTrainer(input_tensor_expected_length, output_tensor_expected_length)
        game_iterator = GameIterator(args.dataset_dir, args.static_board)
        dqn_trainer.train(game_iterator)

    else:
        print(f"The specified path {args.dataset_dir} is not a directory.")


if __name__ == "__main__":
    main()
