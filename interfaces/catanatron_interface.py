import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from environment.board_state import StaticBoardState, DynamicBoardState
from environment.board_state import Node, Edge, HexTile, HexDirection, Port, PortType
from environment.player_state import PlayerState
from environment.action import Action, ActionType
from environment.common import PlayerID, HexTile, ResourceType, BuildingType, Building, Road
from environment.game import CatanGame, GameStep

# DEBUG FLAGS
VERBOSE_LOGGING = False

class CatanatronParser:
    @staticmethod
    def get_action_parameters(action_type: ActionType, static_board_state: StaticBoardState, params: Optional[dict]) -> List[int]:
        parameters = []

        # actions that dont expect any paramaters
        if action_type in {
            ActionType.ROLL,
            ActionType.END_TURN,
            ActionType.DISCARD,
            ActionType.BUY_DEVELOPMENT_CARD,
            ActionType.PLAY_KNIGHT_CARD,
            ActionType.PLAY_ROAD_BUILDING,
        }:
            # No parameters expected for these actions
            if params: raise ValueError(f"No parameters should be provided for action type: {action_type}")
            return parameters

        else:
            # paramaters expected for following actions
            if params is None:
                raise ValueError(f"Parameters cannot be None for action type: {action_type}")

            if action_type == ActionType.MOVE_ROBBER:
                # MOVE_ROBBER params: {"hex_id": int, "target_player_id": int}
                hex_tile = static_board_state.coordinate_to_hex_mapping[tuple(params[0])]
                player_id = PlayerID.string_to_enum(params[1]) if params[1] else -1
                parameters.extend([hex_tile.hex_id, player_id])

            elif action_type == ActionType.BUILD_ROAD:
                # BUILD_ROAD params: {"edge_id": int}
                parameters.extend(tuple(params))
                pass

            elif action_type in {
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY
            }:
                # BUILD_SETTLEMENT/BUILD_CITY params: {"node_id": int}
                parameters.extend([params])

            elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                # PLAY_YEAR_OF_PLENTY params: {"resource_1": int, "resource_2": int}
                if(len(params) == 2):
                    parameters.extend([ResourceType.string_to_enum(params[0]).value,
                                       ResourceType.string_to_enum(params[1]).value])
                else:
                    #TODO(jaisood): Hacky fallback for if there are ever only one value provided for year of plenty
                    print("PLAY_YEAR_OF_PLENTY only one card provided, expected 2")
                    parameters.extend([ResourceType.string_to_enum(params[0]).value,
                                       ResourceType.string_to_enum(params[0]).value])

            elif action_type == ActionType.PLAY_MONOPOLY:
                # PLAY_MONOPOLY params: {"resource": int}
                parameters.extend([ResourceType.string_to_enum(params).value])

            elif action_type == ActionType.MARITIME_TRADE:
                parameters.extend([ResourceType.string_to_enum(param).value if param else -1 for param in params])

            else:
                raise ValueError(f"Unhandled ActionType: {action_type}")

        return parameters

    @staticmethod
    def __parse_board(json_data) -> StaticBoardState:
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
            tile_type = tile['type']
            if "tile_id" in tile:
                hex_id = tile.get("tile_id")
                resource = ResourceType.string_to_enum(tile['resource']) if tile_type == 'RESOURCE_TILE' else ResourceType.DESERT
                number = tile.get("number") if tile_type == 'RESOURCE_TILE' else 0
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
            coordinate_to_hex_mapping={hex_tile.coordinate: hex_tile for hex_tile in hex_tiles},
            id_to_hex_mapping={hex_tile.hex_id: hex_tile for hex_tile in hex_tiles},
            id_to_node_mapping={node.node_id: node for node in nodes},
            id_to_edge_mapping={edge.edge_id: edge for edge in edges},
            id_to_port_mapping=id_to_port_mapping
        )

    @staticmethod
    def parse_board(raw_json_data) -> StaticBoardState:
        json_data = json.loads(raw_json_data)
        return CatanatronParser.__parse_board(json_data)

    @staticmethod
    def parse_board_json(board_path) -> StaticBoardState:
        with open(board_path, 'r') as f:
            json_data = json.load(f)
        return CatanatronParser.__parse_board(json_data)

    @staticmethod
    def __parse_game_step(game_state, static_board_state: StaticBoardState) -> Tuple[List[PlayerState], DynamicBoardState, PlayerID]:
        # Get current player
        current_player = game_state.get('current_player')
        current_player_id = PlayerID.string_to_enum(current_player)
        if VERBOSE_LOGGING: print(f"Current player: {current_player}:{current_player_id}")

        # Parse player states
        player_states = game_state['player_states']
        player_states_list = [
            PlayerState(
                PLAYER_ID=PlayerID.string_to_enum(player_color),
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
            for player_id, player_color in enumerate(game_state['players'])
        ]

        # Parse dynamic board state
        board_state = game_state['board']
        buildings = [
            Building(node_id=building['node_id'],
                        building_type=BuildingType.string_to_enum(building['type']),
                        player_owner=PlayerID.string_to_enum(building['color']))
            for building in board_state['buildings']
        ]

        roads = [
            Road(edge_id=tuple(sorted(road['edge_id'])), player_owner=PlayerID.string_to_enum(road['color']))
            for road in board_state['roads']
        ]

        robber_location = static_board_state.coordinate_to_hex_mapping[tuple(board_state['robber_coordinate'])].hex_id
        if VERBOSE_LOGGING: print(f"Robber location (Hex ID:Coordinate): {robber_location}:{board_state['robber_coordinate']}")
        available_actions = [
            Action(player_id=PlayerID.string_to_enum(player),
                    action=ActionType.string_to_enum(type),
                    parameters = CatanatronParser.get_action_parameters(ActionType.string_to_enum(type), static_board_state, params)
                    )
            for player, type, params in game_state['playable_actions']
        ]

        dynamic_board_state = DynamicBoardState(
            current_player=current_player_id,
            buildings=buildings,
            roads=list(set(roads)),
            robber_location=robber_location,
            available_actions=available_actions
        )

        return (player_states_list, dynamic_board_state, current_player_id)

    @staticmethod
    def parse_data(raw_json_data, static_board_state: StaticBoardState, reorder_players: bool) -> Tuple[List[PlayerState], DynamicBoardState, PlayerID]:
        json_data = json.loads(raw_json_data)
        return CatanatronParser.__parse_game_step(json_data, static_board_state)

    @staticmethod
    def parse_data_json(data_path, static_board_state: StaticBoardState, reorder_players: bool) -> CatanGame:
        with open(data_path, 'r') as f:
            json_data = json.load(f)

        winner = json_data.get('winner')
        winner_id = PlayerID.string_to_enum(winner)
        if(VERBOSE_LOGGING): print(f"Game winner: {winner}:{winner_id}")

        game = json_data.get('game')
        game_steps = []
        for step in game:            
            game_state = step.get('state')
            player_states_list, dynamic_board_state, current_player_id = CatanatronParser.__parse_game_step(game_state, static_board_state)

            if reorder_players:
                reordered_player_states_list = (
                    [player_states_list[game_state['players'].index(winner)]]
                    + [state for i, state in enumerate(player_states_list) if i != game_state['players'].index(winner)]
                )
                assert reordered_player_states_list[0].PLAYER_ID == winner_id, f"ASSERT ERROR: Winning player {winner}:{winner_id} is not the first in the player state array!"
                
                # Replace original players state list with re-ordered one
                player_states_list = reordered_player_states_list

                if VERBOSE_LOGGING:
                    print(f"{winner}: {winner_id}")
                    print(f"Data: {[(player_id, player_color) for player_id, player_color in enumerate(game_state['players'])]}")
                    print(f"Before: {[state.PLAYER_ID for state in player_states_list]}")
                    print(f"Parsed: {[state.PLAYER_ID for state in reordered_player_states_list]}")
                    print("\n")

            # Get action that was taken by player
            action_taken = step.get('action')
            action_taken_by_player = (
                Action(
                    player_id=PlayerID.string_to_enum(action_taken[0]),
                    action=ActionType.string_to_enum(action_taken[1]),
                    parameters = CatanatronParser.get_action_parameters(ActionType.string_to_enum(action_taken[1]), static_board_state, action_taken[2])
                )
                if action_taken
                else Action(
                    player_id=current_player_id,
                    action=ActionType.END_TURN,
                    parameters=[]
                )
            )
            if action_taken is not None: assert action_taken_by_player in dynamic_board_state.available_actions, f"Action: {action_taken} -> {action_taken_by_player} not in {dynamic_board_state.available_actions}: {data_path}"

            game_steps.append(GameStep(step=(player_states_list, dynamic_board_state, action_taken_by_player)))

        return CatanGame(
            winner=winner_id,
            game_steps=game_steps
        )
