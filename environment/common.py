from dataclasses import dataclass
from enum import IntEnum, auto

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
    location: int  # Node ID (0 to 53 on a standard board)
    type: BuildingType
    player_owner: PlayerID