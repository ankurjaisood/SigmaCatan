from abc import ABC, abstractmethod
from typing import Any
from environment import PlayerState, PlayerID, ActionType, Action

class RewardFunction(ABC):
    """
    Abstract base class for defining a reward function.
    """

    def __init__(self,
                 player: PlayerID) -> None:
        self.player = player
        
    @abstractmethod
    #def calculate_reward(self, player_state: Any, action: Any, next_state: Any) -> float:
    def calculate_reward(self, player_state: PlayerState) -> float:
        """
        Calculate the reward for taking an action in a given state
        and transitioning to a next state. FOR NOW ONLY USES PLAYER STATE.

        Args:
            state (Any): The current player state.

        Returns:
            float: The calculated reward.
        """
        assert self.player == player_state.PLAYER_ID, f"Reward function: player ID {self.player} doesnt match ID of player state passed: {player_state.PLAYER_ID}"

class VPRewardFunction(RewardFunction):
    def __init__(self,
                 player: PlayerID):
        super().__init__(player)

    def calculate_reward(self, player_state: PlayerState) -> float:
        super().calculate_reward(player_state)
        return player_state.ACTUAL_VICTORY_POINTS
    
class BasicRewardFunction(RewardFunction):
    def __init__(self,
                 player: PlayerID,
                 w_vp: float = 10.0,
                 w_road: float = 0.5,
                 w_knight: float = float(2/3),
                 w_handSizePenalty: float = 0.1)-> None:
        super().__init__(player)

        self.w_vp = w_vp
        self.w_road = w_road
        self.w_knight = w_knight
        self.w_handSizePenalty = w_handSizePenalty

    def calculate_reward(self, player_state: PlayerState, action: Action) -> float:
        super().calculate_reward(player_state)

        reward = 0.0

        # Reward, victory points
        if action.action == ActionType.BUILD_SETTLEMENT:
            reward += self.w_vp * 1
        elif action.action == ActionType.BUILD_CITY:
            reward += self.w_vp * 2
        else:
            pass
        
        reward += self.w_vp * player_state.ACTUAL_VICTORY_POINTS

        # Reward, board and player state
        if action.action == ActionType.BUILD_ROAD:
            # reward += self.w_road * player_state.LONGEST_ROAD_LENGTH # this uses the longest road length
            reward += self.w_road * (15 - (player_state.ROADS_AVAILABLE - 1) ) # TODO check which road reward is better
        else:
            # reward += self.w_road * player_state.LONGEST_ROAD_LENGTH # this uses the longest road length
            reward += self.w_road * (15 - player_state.ROADS_AVAILABLE)

        reward += self.w_knight * (player_state.KNIGHTS_IN_HAND + player_state.NUMBER_PLAYED_KNIGHT)
        
        total_cards_in_hand = player_state.ORE_IN_HAND + player_state.BRICK_IN_HAND + player_state.WHEAT_IN_HAND + player_state.WOOD_IN_HAND
        reward += self.w_handSizePenalty * min(0, 7 - total_cards_in_hand)

        return reward
