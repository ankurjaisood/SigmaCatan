# environment/game.py

from dataclasses import dataclass
from typing import List, Tuple

from .common import PlayerID
from .player_state import PlayerState
from .board_state import DynamicBoardState
from .action import Action

@dataclass
class GameStep:
    step: Tuple[List[PlayerState], DynamicBoardState]
    action: Action

    def get_player_state_by_ID(self, player_id: PlayerID):
        for p_state in self.step[0]:
            if p_state.PLAYER_ID == player_id:
                return p_state
        return None

@dataclass
class CatanGame:
    winner: PlayerID
    game_steps: List[GameStep]
