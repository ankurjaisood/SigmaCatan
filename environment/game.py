from dataclasses import dataclass
from typing import List, Tuple

from .common import PlayerID
from .player_state import PlayerState
from .board_state import DynamicBoardState
from .action import Action

@dataclass
class CatanGame:
    winner: PlayerID
    game_steps: List[Tuple[List[PlayerState], DynamicBoardState, Action]]
