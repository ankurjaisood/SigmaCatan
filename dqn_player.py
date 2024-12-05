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
from environment.player_state import PlayerState, PlayerID
from environment.action import Action, ActionType

VERBOSE_LOGGING = False

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
        player_states = self.convert_game_state_to_player_state(game.state)
        available_actions = self.convert_playable_actions(playable_actions)
        
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
    def convert_playable_actions(self, playable_actions) -> List[Action]:
        print(playable_actions)
        available_actions = [
                Action(player_id=PlayerID.string_to_enum(player),
                       action=ActionType.string_to_enum(type),
                       parameters = CatanatronParser.get_action_parameters(ActionType.string_to_enum(type), static_board_state, params)
                       )
                for player, type, params in game_state['playable_actions']
            ]
        return available_actions

    def convert_game_state_to_player_state(self, game_state: State) -> List[PlayerState]:
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

        player_states = game_state.player_state
        player_states_list = [
                PlayerState(
                    PLAYER_ID=PlayerID.string_to_enum(player_color.value),
                    HAS_ROLLED=player_states.get(f'P{player_id}_HAS_ROLLED'),
                    HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN=player_states.get(f'P{player_id}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'),
                    VISIBLE_VICTORY_POINTS=player_states.get(f'P{player_id}_VICTORY_POINTS'),
                    ACTUAL_VICTORY_POINTS=player_states.get(f'P{player_id}_ACTUAL_VICTORY_POINTS'),
                    HAS_LONGEST_ROAD=player_states.get(f'P{player_id}_HAS_ROAD'),
                    HAS_LARGEST_ARMY=player_states.get(f'P{player_id}_HAS_ARMY'),
                    ROADS_AVAILABLE=player_states.get(f'P{player_id}_ROADS_AVAILABLE'),
                    SETTLEMENTS_AVAILABLE=player_states.get(f'P{player_id}_SETTLEMENTS_AVAILABLE'),
                    CITIES_AVAILABLE=player_states.get(f'P{player_id}_CITIES_AVAILABLE'),
                    LONGEST_ROAD_LENGTH=player_states.get(f'P{player_id}_LONGEST_ROAD_LENGTH'),
                    WOOD_IN_HAND=player_states.get(f'P{player_id}_WOOD_IN_HAND'),
                    BRICK_IN_HAND=player_states.get(f'P{player_id}_BRICK_IN_HAND'),
                    SHEEP_IN_HAND=player_states.get(f'P{player_id}_SHEEP_IN_HAND'),
                    WHEAT_IN_HAND=player_states.get(f'P{player_id}_WHEAT_IN_HAND'),
                    ORE_IN_HAND=player_states.get(f'P{player_id}_ORE_IN_HAND'),
                    KNIGHTS_IN_HAND=player_states.get(f'P{player_id}_KNIGHT_IN_HAND'),
                    NUMBER_PLAYED_KNIGHT=player_states.get(f'P{player_id}_PLAYED_KNIGHT'),
                    YEAR_OF_PLENTY_IN_HAND=player_states.get(f'P{player_id}_YEAR_OF_PLENTY_IN_HAND'),
                    NUMBER_PLAYED_YEAR_OF_PLENTY=player_states.get(f'P{player_id}_PLAYED_YEAR_OF_PLENTY'),
                    MONOPOLY_IN_HAND=player_states.get(f'P{player_id}_MONOPOLY_IN_HAND'),
                    NUMBER_PLAYED_MONOPOLY=player_states.get(f'P{player_id}_PLAYED_MONOPOLY'),
                    ROAD_BUILDING_IN_HAND=player_states.get(f'P{player_id}_ROAD_BUILDING_IN_HAND'),
                    NUMBER_PLAYED_ROAD_BUILDING=player_states.get(f'P{player_id}_PLAYED_ROAD_BUILDING'),
                    VICTORY_POINT_IN_HAND=player_states.get(f'P{player_id}_VICTORY_POINT_IN_HAND'),
                    NUMBER_PLAYED_VICTORY_POINT=player_states.get(f'P{player_id}_PLAYED_VICTORY_POINT')
                )
                for player_id, player_color in enumerate(game_state.colors)
        ]

        # TODO(jaisood): ADD REORDERING FOR STATES BASED ON FLAG

        if VERBOSE_LOGGING:
            print("OUTPUT")
            print([(player_id, player_color) for player_id, player_color  in enumerate(game_state.colors)])
            print(player_states)
            print(player_states_list)

        # If you have additional mappings from other parts of the state, handle them here
        # For example, resource counts from the board or other players

        return player_states_list