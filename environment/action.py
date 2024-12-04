from enum import IntEnum, auto
from dataclasses import dataclass
from typing import List

from .common import PlayerID, ResourceType

class ActionType(IntEnum):
    # Game over
    GAME_FINISHED = auto()

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
    # OFFER_TRADE = auto()
    # ACCEPT_TRADE = auto()
    # REJECT_TRADE = auto()
    # # CONFIRM_TRADE value is 11-tuple. first 10 as in OFFER_TRADE, last is color of accepting player
    # CONFIRM_TRADE = auto()
    # CANCEL_TRADE = auto()  # value is None

    @staticmethod
    def string_to_enum(s : str):
        return ActionType[s.upper()]

ACTION_TYPE_BASE = {
    ActionType.GAME_FINISHED: 0, # only 1 possible value
    ActionType.ROLL: 1, # only 1 possible value
    ActionType.END_TURN: 2, # only 1 possible value
    ActionType.MOVE_ROBBER: 3, # 19 x 4 possible values
    ActionType.DISCARD: 4, # only 1 possible value
    ActionType.BUILD_ROAD: 5,
    ActionType.BUILD_SETTLEMENT: 6,
    ActionType.BUILD_CITY: 7,
    ActionType.BUY_DEVELOPMENT_CARD: 8,
    ActionType.PLAY_KNIGHT_CARD: 9,
    ActionType.PLAY_YEAR_OF_PLENTY: 10,
    ActionType.PLAY_MONOPOLY: 11,
    ActionType.PLAY_ROAD_BUILDING: 12,
    ActionType.MARITIME_TRADE: 13
}

@dataclass
class Action:
    player_id: PlayerID
    action: ActionType
    parameters: List[int]

    # Static variable to define the maximum number of parameters for any action
    # MAX_ACTION_PARAMETER_LENGTH = 11

    # def flatten(self) -> List[int]:
    #     flattened = [self.action.value, self.player_id.value]

    #     if self.parameters:
    #         flattened.extend(self.parameters)

    #     # Pad with -1 to ensure consistent length
    #     flattened.extend([-1] * (Action.MAX_ACTION_PARAMETER_LENGTH - len(self.parameters)))

    #     if(len(flattened) != Action.MAX_ACTION_PARAMETER_LENGTH + 2):
    #         raise ValueError(
    #             f"Flattened action length mismatch! Expected {Action.MAX_ACTION_PARAMETER_LENGTH + 1}, got {len(flattened)}. "
    #             "Check that MAX_ACTION_PARAMETER_LENGTH is set correctly."
    #         )

    #     return flattened

    # # Works, but too simple:
    # def flatten(self) -> List[int]:
    #     return [self.action.value]

    def flatten(self) -> List[int]:

        ret = [0]
        base = ACTION_TYPE_BASE.get(self.action)
        if base is None:
            raise ValueError(f"Unhandled ActionType: {self.action}")
        
        # TESTING ENUMERATING ACTIONS WITHOUT PARAMETERS
        return [base]

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
            return [base]

        # Ensure parameters are provided
        # Discard shows as NULL parameters in the dataset
        if not self.parameters and self.action != ActionType.DISCARD:
            print (f"Action: {self.action} - Parameters: {self.parameters}, Base: {base}")
            raise ValueError(f"Parameters cannot be empty for action type: {self.action}")

        encoding = 0  # Initialize encoding

        # Mapping based on ActionType
        if self.action == ActionType.MOVE_ROBBER:
            # MOVE_ROBBER params: [hex_id, robber_color_idx]
            if len(self.parameters) < 2:
                raise ValueError(f"MOVE_ROBBER expects at least 2 parameters, got {len(self.parameters)}")
            hex_id, robber_color_idx = self.parameters[:2]
            encoding = hex_id * (len(PlayerID) + 1) + robber_color_idx  # e.g., hex_id * 5 + robber_color_idx
            ret = [base + encoding]

        elif self.action == ActionType.DISCARD:
            # DISCARD params: [resource_idx1, resource_idx2, ...]
            # resources = self.parameters
            # for res in resources:
            #     encoding = encoding * (len(ResourceCardType) + 1) + res  # e.g., res_idx
            # return [base + encoding]
            # Return only base for now, since dataset shows NULL parameters
            ret = [base]

        elif self.action in {ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}:
            # BUILD_ROAD params: [edge_id]
            # BUILD_SETTLEMENT/BUILD_CITY params: [node_id]
            if len(self.parameters) < 1:
                raise ValueError(f"{self.action} expects at least 1 parameter, got {len(self.parameters)}")
            param_id = self.parameters[0]
            encoding = param_id
            ret = [base + encoding]

        elif self.action == ActionType.PLAY_YEAR_OF_PLENTY:
            # PLAY_YEAR_OF_PLENTY params: [resource1_idx, resource2_idx]
            if len(self.parameters) < 2:
                raise ValueError(f"PLAY_YEAR_OF_PLENTY expects at least 2 parameters, got {len(self.parameters)}")
            res1_idx, res2_idx = self.parameters[:2]
            encoding = res1_idx * (len(ResourceType) + 1) + res2_idx
            ret = [base + encoding]

        elif self.action == ActionType.PLAY_MONOPOLY:
            # PLAY_MONOPOLY params: [resource_idx]
            if len(self.parameters) < 1:
                raise ValueError(f"PLAY_MONOPOLY expects at least 1 parameter, got {len(self.parameters)}")
            res_idx = self.parameters[0]
            encoding = res_idx
            ret = [base + encoding]

        elif self.action == ActionType.MARITIME_TRADE:
            # MARITIME_TRADE params: [res1, res2, res3, res4, res5]
            if len(self.parameters) < 5:
                raise ValueError(f"MARITIME_TRADE expects at least 5 parameters, got {len(self.parameters)}")
            resources = self.parameters[:5]
            # Determine resource (A) being traded and count of resources traded (N)
            A = resources[0]
            N = 2
            for i in range(2, 5):
                if resources[i] == A:
                    N += 1
                else:
                    break

            # Determine the resource being acquired (B)
            B = resources[N] if N < 5 else None

            # Encode A (3 bits)
            encoding = A & 0b111

            # Encode N (2 bits) and shift by 3 bits
            encoding |= (N - 2) << 3

            # Encode B (4 bits) and shift by 5 bits
            B_value = 8 if B is None else B
            encoding |= B_value << 5
            ret = [base + encoding]

        else:
            raise ValueError(f"Unhandled ActionType: {self.action}")

        if (ret[0] > (len(ACTION_TYPE_BASE) + 1) * 1000):
            raise ValueError(f"INVALID ENCODING: {ret[0]}, Action: {self.action} - Parameters: {self.parameters}, Base: {base}")

        return ret
