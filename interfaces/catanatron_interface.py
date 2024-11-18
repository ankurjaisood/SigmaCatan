import json
from typing import List, Dict, Tuple
from collections import defaultdict

from environment.board_state import StaticBoardState, DynamicBoardState
from environment.board_state import Node, Edge, HexTile, HexDirection, Port, PortType
from environment.player_state import PlayerState
from environment.action import Action, ActionType
from environment.common import PlayerID, HexTile, ResourceType, BuildingType

# DEBUG FLAGS
VERBOSE_LOGGING = 1

class CatanatronParser:
    @staticmethod
    def parse_board_json(board_path) -> StaticBoardState:
        with open(board_path, 'r') as f:
            json_data = json.load(f)

        hex_tiles = []
        nodes = []
        edges = []
        ports = []

        node_to_hex_mapping = defaultdict(list)
        edge_to_hex_mapping = defaultdict(list)
        port_to_edge_mapping = defaultdict(list)
        id_to_port_mapping = {}

        # Parse hex tiles
        for tile_data in json_data['tiles']:
            tile = tile_data['tile']
            if "tile_id" in tile:
                tile = tile
                hex_id = tile.get("tile_id")
                resource = ResourceType.string_to_enum(tile['resource']) if tile['type'] == 'RESOURCE_TILE' else ResourceType.DESERT
                number = tile.get("number")
                coordinate = tile_data['coordinate']

                hex_tiles.append(HexTile(hex_id=hex_id, resource=resource, token=number, coordinate=tuple(coordinate)))
            elif "port_id" in tile:
                port_id = tile.get('port_id')
                resource = tile.get('resource')
                port_type = PortType.GENERIC if resource is None else PortType[resource]
                
                ports.append(Port(port_id=port_id, port_type=port_type, nodes=list()))
                id_to_port_mapping[port_id] = ports[-1]
            else:
                if VERBOSE_LOGGING: print(f"Tile data not parsed: {tile_data}")

        # Parse nodes
        for node_data in json_data['nodes']:
            node_id = node_data['node_id']
            direction = HexDirection.string_to_enum(node_data['direction'])
            node = Node(node_id=node_id, direction=direction)
            nodes.append(node)

            tile = node_data.get('tile')
            if "tile_id" in tile:
                node_to_hex_mapping[node_id].append(tile.get("tile_id"))
            elif "port_id" in tile:
                id_to_port_mapping[tile.get('port_id')].nodes.append(node)
            else:
                if VERBOSE_LOGGING: print(f"Node data not parsed: {node_data}")

        # Parse edges
        for edge_data in json_data['edges']:
            edge_id = tuple(edge_data['edge_id'])
            direction = HexDirection.string_to_enum(edge_data['direction'])
            edges.append(Edge(edge_id=edge_id, direction=direction))

            tile = edge_data.get('tile')
            if "tile_id" in tile:
                edge_to_hex_mapping[edge_id].append(tile.get("tile_id"))
            elif "port_id" in tile:
                port_to_edge_mapping[tile.get('port_id')].append(edge_id)
            else:
                if VERBOSE_LOGGING: print(f"Edge data not parsed: {edge_data}")

        return StaticBoardState(
            hex_tiles=hex_tiles,
            nodes=nodes,
            edges=edges,
            ports=ports,
            node_to_hex_mapping=node_to_hex_mapping,
            edge_to_hex_mapping=edge_to_hex_mapping,
            port_to_edge_mapping=port_to_edge_mapping,
            coordinate_to_hex_mapping={hex_tile.coordinate for hex_tile in hex_tiles},
            id_to_hex_mapping={hex_tile.hex_id: hex_tile for hex_tile in hex_tiles},
            id_to_node_mapping={node.node_id: node for node in nodes},
            id_to_edge_mapping={edge.edge_id: edge for edge in edges},
            id_to_port_mapping=id_to_port_mapping
        )

    @staticmethod
    def parse_data_json(data_path) -> List[Tuple[List[PlayerState], DynamicBoardState]]:
        with open(data_path, 'r') as f:
            json_data = json.load(f)

        parsed_data = []

        for game_data in json_data:
            game_state = game_data.get('state')
            action_take = game_data.get('action')
            
            # Parse player states
            player_states = [
                PlayerState(
                    **player_data  # Assuming the JSON structure matches PlayerState fields
                )
                for player_data in game_state['players']
            ]

            # Parse dynamic board state
            dynamic_state = game_state['dynamic_board_state']
            buildings = [
                (Node(node_id=building['node']), PlayerID.string_to_enum(building['player']), BuildingType[building['type'].upper()])
                for building in dynamic_state['buildings']
            ]
            roads = [
                (Edge(edge_id=tuple(road['edge'])), PlayerID.string_to_enum(road['player']))
                for road in dynamic_state['roads']
            ]
            robber_location = HexTile(
                hex_id=dynamic_state['robber']['hex_id'],
                resource=None,
                token=None
            )
            available_actions = [
                Action(action=ActionType.string_to_enum(action['action']), parameters=action.get('parameters'))
                for action in dynamic_state['available_actions']
            ]
            dynamic_board_state = DynamicBoardState(
                current_player=PlayerID.string_to_enum(dynamic_state['current_player']),
                buildings=buildings,
                roads=roads,
                robber_location=robber_location,
                available_actions=available_actions
            )

            parsed_data.append((player_states, dynamic_board_state))

        return parsed_data
