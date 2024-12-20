#!/usr/bin/env python3

import os
import argparse
import random
import time
from typing import Generator, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from interfaces.catanatron_interface import CatanatronParser
from environment.board_state import StaticBoardState, DynamicBoardState
from environment.player_state import PlayerState
from environment.game import CatanGame
from environment.action import Action
from rewards.reward_functions import BasicRewardFunction, VPRewardFunction, EndTurnPenaltyRewardFunction
from agents.dqn import DQNTrainer

# DEBUG LOGGING
VERBOSE_LOGGING = False

# INPUT TENSOR SIZE CHECKS
ENABLE_RUNTIME_TENSOR_SIZE_CHECKS = True
FLATTENED_STATIC_BOARD_STATE_LENGTH = 1817
EXPECTED_NUMBER_OF_PLAYERS = 4
FLATTENED_PLAYER_STATE_LENGTH = 26
FLATTENED_DYNAMIC_BOARD_STATE_LENGTH = 486

INPUT_STATE_TENSOR_EXPECTED_LENGTH = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH
INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH
OUTPUT_TENSOR_EXPECTED_LENGTH = 13

FLATTENED_ACTION_LENGTH = 1

class GameIterator:
    def __init__(self,
                 dir_path: str,
                 static_board: bool,
                 disable_dynamic_board_state: bool,
                 reorder_player_states: bool,
                 reward_func_str: str,
                 end_turn_penalty: float) -> None:
        self.static_board = static_board
        self.disable_dynamic_board_state = disable_dynamic_board_state
        self.reorder_player_states = reorder_player_states        
        self.reward_func_str = reward_func_str
        self.end_turn_penalty = end_turn_penalty
        print(f"Using reward function: {self.reward_func_str}, End Turn Penalty: {self.end_turn_penalty}")

        self.parser = CatanatronParser()
        self.games_paths = self.process_directory_iterator(dir_path)
        self.games = self.get_games()
        random.shuffle(self.games)  # Shuffle the games to prevent overfitting
        self.expected_state_tensor_size = (FLATTENED_STATIC_BOARD_STATE_LENGTH if not self.static_board else 0) + \
                                          (EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH) + \
                                          (FLATTENED_DYNAMIC_BOARD_STATE_LENGTH if not self.disable_dynamic_board_state else 0)

    def process_directory_iterator(self, base_dir: str) -> List[Tuple[str, str]]:
        with ThreadPoolExecutor() as executor:
            game_paths = list(executor.map(self.get_game_paths, [os.path.join(root, d) for root, dirs, _ in os.walk(base_dir) for d in dirs]))
        return [game for game in game_paths if game is not None]

    def get_game_paths(self, subdir_path):
        board_file = os.path.join(subdir_path, "board.json")
        data_file = os.path.join(subdir_path, "data.json")
        if os.path.exists(board_file) and os.path.exists(data_file):
            return (board_file, data_file)
        return None

    def parse_data(self, board_path, data_path) -> Tuple[StaticBoardState, CatanGame]:
        static_board_state = self.parser.parse_board_json(board_path)
        game = self.parser.parse_data_json(data_path, static_board_state, self.reorder_player_states)
        return (static_board_state, game)

    def get_games(self) -> List[Tuple[StaticBoardState, CatanGame]]:
        with ThreadPoolExecutor() as executor:
            games = list(executor.map(lambda paths: self.parse_data(*paths), self.games_paths))
        return games

    def create_action_tensor(self, action_taken: Action):
        # Preallocate NumPy array instead of using list
        input_action_tensor = np.array(action_taken.flatten(), dtype=np.float32)
        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            assert input_action_tensor.shape[0] == FLATTENED_ACTION_LENGTH, "Action tensor unexpected size!"
        return input_action_tensor

    def create_state_tensor(self,
                            board_state: StaticBoardState,
                            dynamic_board_state: DynamicBoardState,
                            player_states: List[PlayerState]):
        # Preallocate the state tensor with known sizes to improve performance
        state_tensor = np.zeros(self.expected_state_tensor_size, dtype=np.float32)
        idx = 0

        if not self.static_board:
            board_data = board_state.flatten()
            state_tensor[idx:idx + len(board_data)] = board_data
            idx += len(board_data)

        for player_state in player_states:
            player_data = player_state.flatten()
            state_tensor[idx:idx + len(player_data)] = player_data
            idx += len(player_data)

        if not self.disable_dynamic_board_state:
            dynamic_data = dynamic_board_state.flatten()
            state_tensor[idx:idx + len(dynamic_data)] = dynamic_data

        if ENABLE_RUNTIME_TENSOR_SIZE_CHECKS:
            expected_length = INPUT_STATE_TENSOR_EXPECTED_LENGTH if not self.static_board else INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD
            if self.disable_dynamic_board_state:
                expected_length -= FLATTENED_DYNAMIC_BOARD_STATE_LENGTH
            assert state_tensor.shape[0] == expected_length, f"State tensor unexpected size! {state_tensor.shape[0]}"
        return state_tensor

    def iterate_game(self) -> Generator[list, None, None]:
        static_board_check_state = None
        for game_index, game_data in enumerate(self.games):
            static_board_state, game = game_data
            
            # select reward function
            reward_function = None
            if self.reward_func_str == "VP":
                reward_function = VPRewardFunction(game.winner)
            elif self.reward_func_str == "BASIC":
                reward_function = BasicRewardFunction(game.winner)
            elif self.reward_func_str == "END_TURN_PENALTY":
                reward_function = EndTurnPenaltyRewardFunction(game.winner, w_endTurnPenalty=self.end_turn_penalty)
            else:
                raise ValueError(f"Invalid reward function: {self.reward_func_str}")

            if self.static_board:
                if static_board_check_state is None:
                    static_board_check_state = static_board_state
                else:
                    assert np.array_equal(static_board_check_state.flatten(), static_board_state.flatten()), \
                        f"Static board state is enabled but board state is not constant between games!"

            num_skipped_steps = 0
            for index, step in enumerate(game.game_steps):
                player_states, dynamic_board_state, action_taken = step.step

                # Skip this turn if it's not the winning player's turn
                if game.winner != dynamic_board_state.current_player:
                    num_skipped_steps += 1
                    if VERBOSE_LOGGING: 
                        print(f"Skipping step {index} because it is not the winning player's turn.")
                    continue

                reward = reward_function.calculate_reward(step.get_player_state_by_ID(game.winner), action_taken, dynamic_board_state.available_actions)
                input_state_tensor = self.create_state_tensor(
                    static_board_state,
                    dynamic_board_state,
                    player_states)
                input_action_tensor = self.create_action_tensor(action_taken)

                # Get next state or mark game as finished
                next_state_tensor = None
                if index + 1 < len(game.game_steps):
                    next_player_states, next_dynamic_board_state, _ = game.game_steps[index + 1].step
                    next_state_tensor = self.create_state_tensor(static_board_state, next_dynamic_board_state, next_player_states)
                else:
                    next_state_tensor = np.full(self.expected_state_tensor_size, -1, dtype=np.float32)

                game_finished_tensor = [0] if next_state_tensor is not None else [1]
                yield input_state_tensor, input_action_tensor, [reward], next_state_tensor, game_finished_tensor

            # Print statement after iterating through an entire game
            print(f"Finished iterating game {game_index + 1} out of {len(self.games)}. Number of total steps: {index}. Number of steps as winning player {game.winner.value}: {index - num_skipped_steps}")

    def __iter__(self):
        return self.iterate_game()

def float_between_0_and_1(value):
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")
    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError(f"Value must be between 0 and 1, got {value}")
    return f

def positive_float(value):
    try:
        ivalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be a positive float, got {value}")
    return ivalue

def positive_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be a positive integer, got {value}")
    return ivalue

def main():

    parser = argparse.ArgumentParser(description="Parse Catan board.json and data.json files in subdirectories within a dataset.")
    
    # Unnamed required parameters
    parser.add_argument("dataset_dir", type=str, help="Path to the base directory containing training dataset.")
    
    # Named optional parameters
    parser.add_argument("--static_board", action="store_true", help="Whether to expect a static board for all the games.")
    parser.add_argument("--disable_dynamic_board_state", action="store_true", help="Whether to include the dynamic board state.")
    parser.add_argument("--reorder_players", action="store_true", help="Whether to reorder players such that the winner is always index 0 in the players array.")
    
    # Named required parameters
    parser.add_argument("--reward_func", type=str, required=True, help="Reward function to use. Options: VP, BASIC, END_TURN_PENALTY")
    parser.add_argument("--gamma", type=float_between_0_and_1, required=True, help="Gamma discount factor")
    parser.add_argument("--num_epochs", type=positive_int, required=True, help="Number of epochs to run (positive int)")
    parser.add_argument("--target_update_freq", type=positive_int, required=True, help="Number of steps after which to update target model (positive int)")
    parser.add_argument("--loss_func", type=str, required=True, help="Loss function to use. Options: mse, huber")
    parser.add_argument("--end_turn_penalty", type=positive_float, required=True, help="Reward function penalty for ending turn. Positive float value.")
    parser.add_argument("--tau", type=float_between_0_and_1, required=True, help="Tau value for updating target model. Between 0 and 1.")

    args = parser.parse_args()

    # Process the directory
    if os.path.isdir(args.dataset_dir):        
        if args.static_board:
            input_tensor_expected_length = INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD
        else:
            input_tensor_expected_length = INPUT_STATE_TENSOR_EXPECTED_LENGTH
        
        if args.disable_dynamic_board_state:
            input_tensor_expected_length -= FLATTENED_DYNAMIC_BOARD_STATE_LENGTH

        output_tensor_expected_length = OUTPUT_TENSOR_EXPECTED_LENGTH
        dqn_trainer = DQNTrainer(
            args.reward_func,
            input_tensor_expected_length, 
            output_tensor_expected_length,
            args.gamma,
            args.num_epochs,
            args.target_update_freq,
            args.loss_func,
            args.tau
        )
        
        game_iterator = GameIterator(
            args.dataset_dir, 
            args.static_board, 
            args.disable_dynamic_board_state, 
            args.reorder_players,
            args.reward_func,
            args.end_turn_penalty
        )
        
        start_time = time.time()
        dqn_trainer.train(game_iterator)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Time taken: {duration:.2f} seconds")
    else:
        print(f"The specified path {args.dataset_dir} is not a directory.")


if __name__ == "__main__":
    main()
