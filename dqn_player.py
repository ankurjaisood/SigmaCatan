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

# Enable local SigmaCatan code modules to be imported
module_path = os.path.abspath(os.path.dirname(__file__))  # Adjust as needed
if module_path not in sys.path:
    sys.path.append(module_path)

from interfaces.catanatron_interface import CatanatronParser
from environment.player_state import PlayerState

WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}

@register_player("DQN")
class DQNPlayer(Player):
    """
    Player that decides at random, but skews distribution
    to actions that are likely better (cities > settlements > dev cards).
    """

    def decide(self, game, playable_actions):
        # Convert the game state to PlayerState
        player_state = self.convert_game_state_to_player_state(game.state)
        
        # TODO: Implement DQN-based decision making using player_state
        # For example:
        # action = self.dqn_model.select_action(player_state, playable_actions)
        # return action

        # Placeholder: Existing random choice logic
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight

        )

        return random.choice(bloated_actions)

    def convert_game_state_to_player_state(self, game_state: State) -> PlayerState:
        """
        Converts the overall game state to the PlayerState for the current player.

        Args:
            game_state (State): The current game state.

        Returns:
            PlayerState: The state representation for the current player.

        Raises:
            KeyError: If any expected key is missing in the game_state.player_state.
        """
        current_player_index = game_state.current_player_index
        prefix = f"P{current_player_index}_"
        p_state = game_state.player_state

        # List of all expected keys in PlayerState
        expected_keys = [
            "PLAYER_ID",
            "HAS_ROLLED",
            "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN",
            "VISIBLE_VICTORY_POINTS",
            "ACTUAL_VICTORY_POINTS",
            "HAS_LONGEST_ROAD",
            "HAS_LARGEST_ARMY",
            "ROADS_AVAILABLE",
            "SETTLEMENTS_AVAILABLE",
            "CITIES_AVAILABLE",
            "LONGEST_ROAD_LENGTH",
            "WOOD_IN_HAND",
            "BRICK_IN_HAND",
            "SHEEP_IN_HAND",
            "WHEAT_IN_HAND",
            "ORE_IN_HAND",
            "KNIGHTS_IN_HAND",
            "NUMBER_PLAYED_KNIGHT",
            "YEAR_OF_PLENTY_IN_HAND",
            "NUMBER_PLAYED_YEAR_OF_PLENTY",
            "MONOPOLY_IN_HAND",
            "NUMBER_PLAYED_MONOPOLY",
            "ROAD_BUILDING_IN_HAND",
            "NUMBER_PLAYED_ROAD_BUILDING",
            "VICTORY_POINT_IN_HAND",
            "NUMBER_PLAYED_VICTORY_POINT",
        ]

        # Extract values, ensuring all keys are present
        player_state_values = {}
        missing_keys = []
        for key in expected_keys:
            full_key = prefix + key
            if full_key in p_state:
                player_state_values[key] = p_state[full_key]
            else:
                missing_keys.append(full_key)

        if missing_keys:
            raise KeyError(
                f"Missing keys in game_state.player_state for player index {current_player_index}: {missing_keys}"
            )

        # Construct the PlayerState object
        player_state = PlayerState(
            PLAYER_ID=player_state_values["PLAYER_ID"],
            HAS_ROLLED=player_state_values["HAS_ROLLED"],
            HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN=player_state_values["HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
            
            VISIBLE_VICTORY_POINTS=player_state_values["VISIBLE_VICTORY_POINTS"],
            ACTUAL_VICTORY_POINTS=player_state_values["ACTUAL_VICTORY_POINTS"],
            HAS_LONGEST_ROAD=player_state_values["HAS_LONGEST_ROAD"],
            HAS_LARGEST_ARMY=player_state_values["HAS_LARGEST_ARMY"],
            
            ROADS_AVAILABLE=player_state_values["ROADS_AVAILABLE"],
            SETTLEMENTS_AVAILABLE=player_state_values["SETTLEMENTS_AVAILABLE"],
            CITIES_AVAILABLE=player_state_values["CITIES_AVAILABLE"],
            LONGEST_ROAD_LENGTH=player_state_values["LONGEST_ROAD_LENGTH"],
            
            WOOD_IN_HAND=player_state_values["WOOD_IN_HAND"],
            BRICK_IN_HAND=player_state_values["BRICK_IN_HAND"],
            SHEEP_IN_HAND=player_state_values["SHEEP_IN_HAND"],
            WHEAT_IN_HAND=player_state_values["WHEAT_IN_HAND"],
            ORE_IN_HAND=player_state_values["ORE_IN_HAND"],
            
            KNIGHTS_IN_HAND=player_state_values["KNIGHTS_IN_HAND"],
            NUMBER_PLAYED_KNIGHT=player_state_values["NUMBER_PLAYED_KNIGHT"],
            YEAR_OF_PLENTY_IN_HAND=player_state_values["YEAR_OF_PLENTY_IN_HAND"],
            NUMBER_PLAYED_YEAR_OF_PLENTY=player_state_values["NUMBER_PLAYED_YEAR_OF_PLENTY"],
            MONOPOLY_IN_HAND=player_state_values["MONOPOLY_IN_HAND"],
            NUMBER_PLAYED_MONOPOLY=player_state_values["NUMBER_PLAYED_MONOPOLY"],
            ROAD_BUILDING_IN_HAND=player_state_values["ROAD_BUILDING_IN_HAND"],
            NUMBER_PLAYED_ROAD_BUILDING=player_state_values["NUMBER_PLAYED_ROAD_BUILDING"],
            VICTORY_POINT_IN_HAND=player_state_values["VICTORY_POINT_IN_HAND"],
            NUMBER_PLAYED_VICTORY_POINT=player_state_values["NUMBER_PLAYED_VICTORY_POINT"],
        )

        # If you have additional mappings from other parts of the state, handle them here
        # For example, resource counts from the board or other players

        return player_state