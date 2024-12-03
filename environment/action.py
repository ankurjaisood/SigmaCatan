# action.py

from enum import IntEnum, auto
from dataclasses import dataclass
from typing import List, Optional
from .common import PlayerID, ResourceCardType  # Importing from common.py

# Define ActionType Enum with excluded trade actions
class ActionType(IntEnum):
    # Game Over
    GAME_FINISHED = auto()

    # Discrete Actions
    ROLL = auto()  # value is None
    END_TURN = auto()  # value is None

    # Move Actions
    MOVE_ROBBER = auto()  # value is (hex_id, robber_color)
    DISCARD = auto()  # value is resources (list)

    # Building/Buying Actions
    BUILD_ROAD = auto()  # value is edge_id
    BUILD_SETTLEMENT = auto()  # value is node_id
    BUILD_CITY = auto()  # value is node_id
    BUY_DEVELOPMENT_CARD = auto()  # value is None

    # Development Cards
    PLAY_KNIGHT_CARD = auto()  # value is None
    PLAY_YEAR_OF_PLENTY = auto()  # value is (resource1, resource2)
    PLAY_MONOPOLY = auto()  # value is resource
    PLAY_ROAD_BUILDING = auto()  # value is None

    # MARITIME_TRADE value is 5-resouce tuple, where last resource is resource asked.
    #   resources in index 2 and 3 might be None, denoting a port-trade.
    MARITIME_TRADE = auto()

    # Excluded Trade Actions
    # OFFER_TRADE = auto()
    # ACCEPT_TRADE = auto()
    # REJECT_TRADE = auto()
    # CONFIRM_TRADE = auto()
    # CANCEL_TRADE = auto()

    @staticmethod
    def string_to_enum(s: str):
        try:
            return ActionType[s.upper()]
        except KeyError:
            raise ValueError(f"Invalid ActionType string: {s}")

# Assign a unique base index to each ActionType
ACTION_TYPE_BASE = {
    ActionType.GAME_FINISHED: 0,
    ActionType.ROLL: 1000,
    ActionType.END_TURN: 2000,
    ActionType.MOVE_ROBBER: 3000,
    ActionType.DISCARD: 4000,
    ActionType.BUILD_ROAD: 5000,
    ActionType.BUILD_SETTLEMENT: 6000,
    ActionType.BUILD_CITY: 7000,
    ActionType.BUY_DEVELOPMENT_CARD: 8000,
    ActionType.PLAY_KNIGHT_CARD: 9000,
    ActionType.PLAY_YEAR_OF_PLENTY: 10000,
    ActionType.PLAY_MONOPOLY: 11000,
    ActionType.PLAY_ROAD_BUILDING: 12000,
    ActionType.MARITIME_TRADE: 13000
    # Trade Actions Removed
    # ActionType.OFFER_TRADE: 14000,
    # ActionType.ACCEPT_TRADE: 15000,
    # ActionType.REJECT_TRADE: 16000,
    # ActionType.CONFIRM_TRADE: 17000,
    # ActionType.CANCEL_TRADE: 18000,
}

@dataclass
class Action:
    player_id: PlayerID
    action: ActionType
    parameters: List[int]

    # Static variable to define the maximum number of parameters for any action
    MAX_ACTION_PARAMETER_LENGTH = 11

    def flatten(self) -> int:
        """
        Convert the Action instance into a unique integer based on ActionType and parameters.
        """
        # Start with the base index for the ActionType
        base = ACTION_TYPE_BASE.get(self.action)
        if base is None:
            raise ValueError(f"Unhandled ActionType: {self.action}")

        # Actions without parameters
        if self.action in {
            ActionType.GAME_FINISHED,
            ActionType.ROLL,
            ActionType.END_TURN,
            ActionType.BUY_DEVELOPMENT_CARD,
            ActionType.PLAY_KNIGHT_CARD,
            ActionType.PLAY_ROAD_BUILDING
        }:
            if self.parameters:
                raise ValueError(f"No parameters should be provided for action type: {self.action}")
            return base

        # Ensure parameters are provided
        if not self.parameters:
            raise ValueError(f"Parameters cannot be empty for action type: {self.action}")

        encoding = 0  # Initialize encoding

        # Mapping based on ActionType
        if self.action == ActionType.MOVE_ROBBER:
            # MOVE_ROBBER params: [hex_id, robber_color_idx]
            if len(self.parameters) < 2:
                raise ValueError(f"MOVE_ROBBER expects at least 2 parameters, got {len(self.parameters)}")
            hex_id, robber_color_idx = self.parameters[:2]
            encoding = hex_id * (len(PlayerID) + 1) + robber_color_idx  # e.g., hex_id * 5 + robber_color_idx
            return base + encoding

        elif self.action == ActionType.DISCARD:
            # DISCARD params: [resource_idx1, resource_idx2, ...]
            resources = self.parameters
            for res in resources:
                encoding = encoding * (len(ResourceCardType) + 1) + res  # e.g., res_idx
            return base + encoding

        elif self.action in {ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}:
            # BUILD_ROAD params: [edge_id]
            # BUILD_SETTLEMENT/BUILD_CITY params: [node_id]
            if len(self.parameters) < 1:
                raise ValueError(f"{self.action} expects at least 1 parameter, got {len(self.parameters)}")
            param_id = self.parameters[0]
            encoding = param_id
            return base + encoding

        elif self.action == ActionType.PLAY_YEAR_OF_PLENTY:
            # PLAY_YEAR_OF_PLENTY params: [resource1_idx, resource2_idx]
            if len(self.parameters) < 2:
                raise ValueError(f"PLAY_YEAR_OF_PLENTY expects at least 2 parameters, got {len(self.parameters)}")
            res1_idx, res2_idx = self.parameters[:2]
            encoding = res1_idx * (len(ResourceCardType) + 1) + res2_idx
            return base + encoding

        elif self.action == ActionType.PLAY_MONOPOLY:
            # PLAY_MONOPOLY params: [resource_idx]
            if len(self.parameters) < 1:
                raise ValueError(f"PLAY_MONOPOLY expects at least 1 parameter, got {len(self.parameters)}")
            res_idx = self.parameters[0]
            encoding = res_idx
            return base + encoding

        elif self.action == ActionType.MARITIME_TRADE:
            # MARITIME_TRADE params: [res1, res2, res3, res4, res5]
            if len(self.parameters) < 5:
                raise ValueError(f"MARITIME_TRADE expects at least 5 parameters, got {len(self.parameters)}")
            resources = self.parameters[:5]
            for res in resources:
                # Use len(ResourceCardType) to represent None (if res is None, it should be encoded as len(ResourceCardType))
                res_encoded = res if res != -1 else len(ResourceCardType)
                encoding = encoding * (len(ResourceCardType) + 1) + res_encoded
            return base + encoding

        else:
            raise ValueError(f"Unhandled ActionType: {self.action}")

        # Pad parameters with -1 to ensure consistent length (if necessary)
        # Not needed here since we return an integer
