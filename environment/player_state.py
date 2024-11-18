from dataclasses import dataclass

from .common import PlayerID

'''
Custom definition of a players state in Cata
'''
@dataclass
class PlayerState:
    # Player turn info
    PLAYER_ID: PlayerID
    HAS_ROLLED: bool
    HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN: bool

    # Victory point info
    VISIBLE_VICTORY_POINTS: int
    ACTUAL_VICTORY_POINTS: int
    HAS_LONGEST_ROAD: bool
    HAS_LARGEST_ARMY: bool

    # Building info
    ROADS_AVAILABLE: int
    SETTLEMENTS_AVAILABLE: int
    CITIES_AVAILABLE: int
    LONGEST_ROAD_LENGTH: int

    # Player hand info    
    WOOD_IN_HAND: int
    BRICK_IN_HAND: int
    SHEEP_IN_HAND: int
    WHEAT_IN_HAND: int
    ORE_IN_HAND: int
    
    # Development card info
    KNIGHTS_IN_HAND: int
    NUMBER_PLAYED_KNIGHT: int
    YEAR_OF_PLENTY_IN_HAND: int
    NUMBER_PLAYED_YEAR_OF_PLENTY: int
    MONOPOLY_IN_HAND: int
    NUMBER_PLAYED_MONOPOLY: int
    ROAD_BUILDING_IN_HAND: int
    NUMBER_PLAYED_ROAD_BUILDING: int
    VICTORY_POINT_IN_HAND: int
    NUMBER_PLAYED_VICTORY_POINT: int

    '''
    Flattens the state into an array which can be used to create input tensor for network

    Converts booleans to ints [0,1]
    '''
    def flatten(self):
        return [
            self.PLAYER_ID.value,
            int(self.HAS_ROLLED),
            int(self.HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN),
            self.VISIBLE_VICTORY_POINTS,
            self.ACTUAL_VICTORY_POINTS,
            int(self.HAS_LONGEST_ROAD),
            int(self.HAS_LARGEST_ARMY),
            self.ROADS_AVAILABLE,
            self.SETTLEMENTS_AVAILABLE,
            self.CITIES_AVAILABLE,
            self.LONGEST_ROAD_LENGTH,
            self.WOOD_IN_HAND,
            self.BRICK_IN_HAND,
            self.SHEEP_IN_HAND,
            self.WHEAT_IN_HAND,
            self.ORE_IN_HAND,
            self.KNIGHTS_IN_HAND,
            self.NUMBER_PLAYED_KNIGHT,
            self.YEAR_OF_PLENTY_IN_HAND,
            self.NUMBER_PLAYED_YEAR_OF_PLENTY,
            self.MONOPOLY_IN_HAND,
            self.NUMBER_PLAYED_MONOPOLY,
            self.ROAD_BUILDING_IN_HAND,
            self.NUMBER_PLAYED_ROAD_BUILDING,
            self.VICTORY_POINT_IN_HAND,
            self.NUMBER_PLAYED_VICTORY_POINT,
        ]
