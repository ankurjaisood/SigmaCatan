# test_catanatron_parser.py

import unittest
from interfaces.catanatron_interface import CatanatronParser
from environment.common import PlayerID
from environment.player_state import PlayerState
from environment.board_state import StaticBoardState, DynamicBoardState
from environment.game import GameStep, CatanGame
from environment.action import Action

class TestCatanatronParser(unittest.TestCase):
    def test_parse_data_json(self):
        parser = CatanatronParser()
        static_board_state = parser.parse_board_json("datasets/2024-11-17_20_48_48/0a6c0574-3534-4f57-a802-a000ed8407ad/board.json")
        catan_game = parser.parse_data_json("datasets/2024-11-17_20_48_48/0a6c0574-3534-4f57-a802-a000ed8407ad/data.json", static_board_state)

        for index, step in enumerate(catan_game.game_steps):
            with self.subTest(step=index):
                self.assertIsInstance(step, GameStep)
                self.assertIsInstance(step.step, tuple)
                self.assertEqual(len(step.step), 2, f"Step {index} does not have 2 elements in step tuple.")

                player_states, dynamic_board_state = step.step

                # Validate player_states
                self.assertIsInstance(player_states, list)
                for ps in player_states:
                    self.assertIsInstance(ps, PlayerState)

                # Validate dynamic_board_state
                self.assertIsInstance(dynamic_board_state, DynamicBoardState)
                self.assertIsInstance(dynamic_board_state.robber_location, int)
                self.assertIsInstance(dynamic_board_state.available_actions, list)
                for action in dynamic_board_state.available_actions:
                    self.assertIsInstance(action, Action)

                # Validate action_taken_by_player
                self.assertIsInstance(step.action, Action)

if __name__ == '__main__':
    unittest.main()
