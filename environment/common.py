from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Tuple

class PlayerID(IntEnum):
    ORANGE = auto()
    WHITE = auto()
    BLUE = auto()
    RED = auto()

    @staticmethod
    def string_to_enum(s : str):
        return PlayerID[s.upper()]

class ResourceType(IntEnum):
    WATER = auto()
    DESERT = auto()
    WOOD = auto()
    BRICK = auto()
    SHEEP = auto()
    WHEAT = auto()
    ORE = auto()

    @staticmethod
    def string_to_enum(s : str):
        return ResourceType[s.upper()]

class BuildingType(IntEnum):
    SETTLEMENT = auto()
    CITY = auto()
    ROAD = auto()

    @staticmethod
    def string_to_enum(s : str):
        return BuildingType[s.upper()]

@dataclass
class HexTile:
    hex_id: int  # Hex ID (0-18 for standard board)
    resource: ResourceType
    token: int  # Dice roll number (2-12), or None for the desert
    coordinate: tuple

@dataclass
class Building:
    node_id: int  # Node ID (0 to 53 on a standard board)
    building_type: BuildingType
    player_owner: PlayerID

    def flatten(self):
        return [self.node_id, self.building_type.value, self.player_owner.value]

@dataclass
class Road:
    edge_id: Tuple[int, int]
    player_owner: PlayerID

    def flatten(self):
        return [self.edge_id[0], self.edge_id[1], self.player_owner.value]

    def __hash__(self):
        return hash((self.edge_id, self.player_owner))

    def __eq__(self, other):
        if not isinstance(other, Road):
            return False
        return (
            self.edge_id == other.edge_id and
            self.player_owner == other.player_owner
        )

