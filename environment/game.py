from dataclasses import dataclass
from typing import List, Tuple

from .common import PlayerID
from .player_state import PlayerState
from .board_state import DynamicBoardState
from .action import Action

@dataclass
class GameStep:
    step: Tuple[List[PlayerState], DynamicBoardState, Action]

    def get_player_state_by_ID(self, player_id: PlayerID):
        player_state = None
        for p_sate in self.step[0]:
            if p_sate.PLAYER_ID == player_id:
                player_state = p_sate
                break
        return player_state

@dataclass
class CatanGame:
    winner: PlayerID
    game_steps: List[GameStep]
