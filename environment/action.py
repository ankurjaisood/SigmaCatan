from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional, List

from .common import PlayerID

class ActionType(IntEnum):
    # Discrete Actions
    ROLL = auto()  # value is None
    END_TURN = auto()  # value is None

    MOVE_ROBBER = auto()  # value is (coordinate -> hex ID, Color|None)
    DISCARD = auto()  # value is None|Resource[]

    # Building/Buying
    BUILD_ROAD = auto()  # value is edge_id
    BUILD_SETTLEMENT = auto()  # value is node_id
    BUILD_CITY = auto()  # value is node_id
    BUY_DEVELOPMENT_CARD = auto()  # value is None

    # Development Cards
    PLAY_KNIGHT_CARD = auto()  # value is None
    PLAY_YEAR_OF_PLENTY = auto()  # value is (Resource, Resource)
    PLAY_MONOPOLY = auto()  # value is Resource
    PLAY_ROAD_BUILDING = auto()  # value is None

    # Trading out of scope for now

    # MARITIME_TRADE value is 5-resouce tuple, where last resource is resource asked.
    #   resources in index 2 and 3 might be None, denoting a port-trade.
    MARITIME_TRADE = auto()
    # Domestic Trade (player to player trade)
    # Values for all three is a 10-resource tuple, first 5 is offered freqdeck, last 5 is
    #   receiving freqdeck.
    OFFER_TRADE = auto()
    ACCEPT_TRADE = auto()
    REJECT_TRADE = auto()
    # CONFIRM_TRADE value is 11-tuple. first 10 as in OFFER_TRADE, last is color of accepting player
    CONFIRM_TRADE = auto()
    CANCEL_TRADE = auto()  # value is None

    @staticmethod
    def string_to_enum(s : str):
        return ActionType[s.upper()]
    
@dataclass
class Action:
    player_id: PlayerID
    action: ActionType
    parameters: Optional[List[int]] = None

    # Static variable to define the maximum number of parameters for any action
    MAX_ACTION_PARAMETER_LENGTH = 11

    def flatten(self) -> List[int]:
        flattened = [int(self.action)]
        if self.parameters:
            flattened.extend(self.parameters)

        # Pad with -1 to ensure consistent length
        flattened.extend([-1] * (Action.MAX_ACTION_PARAMETER_LENGTH - len(self.parameters or [])))

        if(len(flattened) != Action.MAX_ACTION_PARAMETER_LENGTH + 1):
            raise ValueError(
                f"Flattened action length mismatch! Expected {Action.MAX_ACTION_PARAMETER_LENGTH + 1}, got {len(flattened)}. "
                "Check that MAX_ACTION_PARAMETER_LENGTH is set correctly."
            )

        return flattened
