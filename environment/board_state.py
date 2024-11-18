from enum import IntEnum, auto
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .common import HexTile, PlayerID, BuildingType
from .action import Action

class HexDirection(IntEnum):
    NORTH = auto()
    NORTHEAST = auto()
    EAST = auto()
    SOUTHEAST = auto()
    SOUTH = auto()
    SOUTHWEST = auto()
    WEST = auto()
    NORTHWEST = auto()
    
    @staticmethod
    def string_to_enum(s : str):
        return HexDirection[s.upper()]

class PortType(IntEnum):
    GENERIC = auto() # 3:1
    WOOD = auto() # 2:1 Wood
    BRICK = auto() # 2:1 Brick
    SHEEP = auto() # 2:1 Sheep
    WHEAT = auto() # 2:1 Wheat
    ORE = auto() # 2:1 Ore

@dataclass
class Node:
    node_id: int
    direction: HexDirection

@dataclass
class Edge:
    edge_id: Tuple[int, int]
    direction: HexDirection

@dataclass
class Port:
    port_id: int
    port_type: PortType
    nodes: List[Node]

@dataclass
class StaticBoardState:
    hex_tiles: List[HexTile]
    nodes: List[Node]
    edges: List[Edge]
    ports: List[Port]

    # Mappings
    node_to_hex_mapping: Dict[int, List[int]]  # Node ID -> List of Hex IDs
    edge_to_hex_mapping: Dict[Tuple[int, int], List[int]]  # Edge (node_a, node_b) -> List of Hex IDs
    port_to_edge_mapping: Dict[int, Tuple[int, int]]  # Port ID -> Edge (node_a, node_b)

    # Helper mappings, not flattened
    coordinate_to_hex_mapping: Dict[Tuple[int, int, int], int] # x,y,z -> Hex ID
    id_to_hex_mapping: Dict[int, HexTile]
    id_to_node_mapping: Dict[int, Node]
    id_to_edge_mapping: Dict[Tuple[int, int], Edge]
    id_to_port_mapping: Dict[int, Port]

    def flatten(self):
        flattened = []

        # Flatten hex_tiles
        for hex_tile in self.hex_tiles:
            flattened.extend([
                hex_tile.resource.value,  # Convert resource enum to int
                hex_tile.token  # Number token
            ])

        # Flatten nodes
        for node in self.nodes:
            flattened.extend([
                node.node_id,
                node.direction
            ])

        # Flatten edges
        for edge in self.edges:
            flattened.extend([
                edge.edge_id,
                edge.direction
            ])

        # Flatten ports
        for port in self.ports:
            flattened.extend([
                port.port_id,
                port.port_type,
                *port.nodes  # Add the two nodes associated with the port
            ])

        # Flatten node_to_hex_mapping
        for node_id, hex_ids in self.node_to_hex_mapping.items():
            flattened.append(node_id)
            flattened.extend(hex_ids)

        # Flatten edge_to_hex_mapping
        for (node_a, node_b), hex_ids in self.edge_to_hex_mapping.items():
            flattened.extend([node_a, node_b])
            flattened.extend(hex_ids)

        # Flatten port_to_edge_mapping
        for port_id, (node_a, node_b) in self.port_to_edge_mapping.items():
            flattened.append(port_id)
            flattened.extend([node_a, node_b])

        return flattened

@dataclass
class DynamicBoardState:
    current_player: PlayerID
    buildings: List[Tuple[Node, PlayerID, BuildingType]]
    roads: List[Tuple[Edge, PlayerID]]
    robber_location: HexTile
    available_actions: List[Action]

    MAX_NUM_PLACED_BUILDINGS = 80 # 4 players * 5 settlements * 4 cities
    MAX_NUM_PLACED_ROADS = 60 # 4 players * 15 roads
    MAX_NUM_AVAILABLE_ACTIONS = 64

    def flatten(self) -> List[int]:
        flattened = []

        # Flatten current player
        flattened.append(self.current_player)

        # Flatten buildings
        for building in self.buildings:
            flattened.extend(building)  # Node ID, PlayerID, BuildingType
        # Pad to MAX_NUM_PLACED_BUILDINGS
        flattened.extend([-1] * ((self.MAX_NUM_PLACED_BUILDINGS * 3) - len(self.buildings) * 3))

        # Flatten roads
        for road in self.roads:
            flattened.extend(road)  # Edge ID, PlayerID
        # Pad to MAX_NUM_PLACED_ROADS
        flattened.extend([-1] * ((self.MAX_NUM_PLACED_ROADS * 2) - len(self.roads) * 2))

        # Flatten robber location
        flattened.append(self.robber_location)

        # Flatten available actions
        for action in self.available_actions:
            flattened.extend(action.flatten())
        # Pad to MAX_NUM_AVAILABLE_ACTIONS
        flattened.extend(
            [-1] * ((self.MAX_NUM_AVAILABLE_ACTIONS * (Action.MAX_ACTION_PARAMETER_LENGTH + 1)) - 
                    len(self.available_actions) * (Action.MAX_ACTION_PARAMETER_LENGTH + 1))
        )

        return flattened
