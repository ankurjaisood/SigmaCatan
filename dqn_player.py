# dqn_player.py
import random
import os
import sys
from pprint import pprint
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player
from catanatron.models.actions import ActionType
from catanatron.state import State
import torch

# Enable local SigmaCatan code modules to be imported
module_path = os.path.abspath(os.path.dirname(__file__))  # Adjust as needed
if module_path not in sys.path:
    sys.path.append(module_path)

import json
import numpy as np
from catanatron_experimental.cli.accumulators import SigmaCatanDataAccumulator
from interfaces.catanatron_interface import CatanatronParser
from environment.player_state import PlayerState, PlayerID
from environment.action import Action, ActionType
from environment.board_state import StaticBoardState, DynamicBoardState
from environment.player_state import PlayerState
from agents.dqn import DQN
from main import INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD, OUTPUT_TENSOR_EXPECTED_LENGTH

VERBOSE_LOGGING = False

WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}

# INPUT TENSOR SIZE CHECKS
ENABLE_RUNTIME_TENSOR_SIZE_CHECKS = True
FLATTENED_STATIC_BOARD_STATE_LENGTH = 1817
EXPECTED_NUMBER_OF_PLAYERS = 4
FLATTENED_PLAYER_STATE_LENGTH = 26
FLATTENED_DYNAMIC_BOARD_STATE_LENGTH = 486

INPUT_STATE_TENSOR_EXPECTED_LENGTH = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + FLATTENED_STATIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH
INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD = FLATTENED_DYNAMIC_BOARD_STATE_LENGTH + EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH
OUTPUT_TENSOR_EXPECTED_LENGTH = 14

FLATTENED_ACTION_LENGTH = 1

#MODEL_PATH = "./models/static_board/model-20241204_122518-590x14:302-gamma_0.99-lr_0.0001-bs_512-epochs_1-updatefreq_2500.pth"
MODEL_PATH = "./models/static_board/model-20241205_211410-590x14:302-gamma_0.99-lr_0.0001-bs_512-epochs_5-updatefreq_5000.pth"

@register_player("DQN")
class DQNPlayer(Player):
    def __init__(self, color, is_bot=True):
        super().__init__(color, is_bot)
        self.parser = CatanatronParser()
        self.indent = "\t"

        ## TODO(jai): CODE DUPLICATION FROM main.py
        self.static_board = True # Move to constructor
        self.disable_dynamic_board_state = False # Move to constructor
        self.expected_state_tensor_size = (FLATTENED_STATIC_BOARD_STATE_LENGTH if not self.static_board else 0) + \
                                    (EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH) + \
                                    (FLATTENED_DYNAMIC_BOARD_STATE_LENGTH if not self.disable_dynamic_board_state else 0)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PYTORCH USING DEVICE {self.device}")
        self.model_path = MODEL_PATH
        self.model = self._load_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.step_counter = 0
        self.invalid_step_counter = 0

    def __del__(self):
        print(f"Invalid actions/total actions: {self.invalid_step_counter}/{self.step_counter}")


    def _load_model(self, model_path):
        # Replace DQNModel with your actual model class
        model = DQN(INPUT_STATE_TENSOR_EXPECTED_LENGTH_STATIC_BOARD, OUTPUT_TENSOR_EXPECTED_LENGTH, hidden_layer_size=302)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
        return model

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

    def decide(self, game, playable_actions):
        static_board_json = json.dumps(game.state.board.map, cls=SigmaCatanDataAccumulator.SigmaCatanGameEncoder, indent=self.indent)
        static_board = self.parser.parse_board(static_board_json)

        game_data_json = json.dumps(game.state, cls=SigmaCatanDataAccumulator.SigmaCatanGameEncoder, indent=self.indent)
        player_states, dynamic_board_state, player_id = self.parser.parse_data(game_data_json, static_board, False)

        input_state_tensor = self.create_state_tensor(static_board, dynamic_board_state, player_states)
        output_tensor = self.model.forward(torch.from_numpy(input_state_tensor))

        best_action_idx = torch.argmax(output_tensor).item()
        best_action_idx += 1 # TODO(jaisood): PYTHON ENUMS START FROM 1 WHEN YOU USE auto()
        best_action = ActionType(best_action_idx)

        if VERBOSE_LOGGING: print(f"Best action idx: {best_action_idx}, Best Action: {best_action.name}")

        allowable_action_list = [ActionType.string_to_enum(action.action_type.value) for action in playable_actions]
        self.step_counter += 1
        if(best_action in allowable_action_list and best_action != ActionType.END_TURN):
            selected_allowable_actions = [(idx, action) for idx, action in enumerate(allowable_action_list) if action == best_action]
            if VERBOSE_LOGGING: print(selected_allowable_actions)
            selected_action = random.choice(selected_allowable_actions)
            return playable_actions[selected_action[0]]
        else:
            self.invalid_step_counter += 1
            return random.choice(playable_actions)
