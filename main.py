# main.py

import json
from typing import Iterator, Tuple, List
import logging

from interfaces.catanatron_interface import CatanatronParser
from environment.board_state import StaticBoardState, DynamicBoardState
from environment.game import CatanGame, GameStep
from environment.action import Action, ActionType
from environment.player_state import PlayerState
from environment.common import PlayerID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (Adjust these based on your actual configuration)
INPUT_STATE_TENSOR_EXPECTED_LENGTH = 128  # Example value; adjust as needed


def create_action_from_game_step(step_data: dict, static_board_state: StaticBoardState) -> Action:
    """
    Creates an Action object from game step data.

    Args:
        step_data (dict): Dictionary containing state and action information.
        static_board_state (StaticBoardState): The static board state.

    Returns:
        Action: An Action object representing the action taken.
    """
    try:
        # Unpack the action list
        player_id_str, action_type_str, *parameters = step_data['action']
    except ValueError:
        logger.error("Action data is malformed. Expected at least two elements.")
        return None

    try:
        # Corrected variable name from 'action_id_str' to 'player_id_str'
        player_id = PlayerID[player_id_str.upper()]
    except KeyError:
        logger.error(f"Invalid player ID string: {player_id_str}")
        return None

    try:
        action_type = ActionType[action_type_str.upper()]
    except KeyError:
        logger.error(f"Invalid action type string: {action_type_str}")
        return None

    return Action(player_id=player_id, action=action_type, parameters=parameters)


class RewardFunction:
    """
    Placeholder for the reward function.
    Implement your reward calculation logic here.
    """
    def calculate_reward(self, player_state: PlayerState, action_list: List[str]) -> float:
        # Example implementation (to be replaced with actual logic)
        return 1.0  # Reward value


class StateTensorCreator:
    """
    Placeholder for the state tensor creation.
    Implement your state tensor creation logic here.
    """
    def create_state_tensor(self, static_board_state: StaticBoardState, dynamic_board_state_dict: dict, player_states_dict: List[dict]) -> List[float]:
        # Example implementation (to be replaced with actual logic)
        return [0.0] * INPUT_STATE_TENSOR_EXPECTED_LENGTH  # Dummy tensor


class GameIterator:
    def __init__(self, game: CatanGame, static_board_state: StaticBoardState):
        self.game = game
        self.static_board_state = static_board_state
        self.reward_function = RewardFunction()
        self.create_state_tensor = StateTensorCreator().create_state_tensor

    def iterate_game(self) -> Iterator[Tuple[List[float], int, float, List[float], int]]:
        for index, step in enumerate(self.game.game_steps):
            # Enhanced Debugging: Print detailed step information
            logger.debug(f"\nProcessing Step {index}:")
            logger.debug(f"step.step Type: {type(step.step)}")
            logger.debug(f"step.step Content: {step.step}")
            logger.debug(f"step.action Type: {type(step.action)}")
            logger.debug(f"step.action Content: {step.action}")

            # Ensure step is a GameStep instance
            if not isinstance(step, GameStep):
                logger.error(f"Unexpected step format at step {index}: {step}")
                continue

            # Unpack the step tuple
            step_tuple = step.step

            # Validate the tuple structure
            if not isinstance(step_tuple, tuple):
                logger.error(f"Unexpected step.step type at step {index}: {type(step_tuple)}. Expected tuple.")
                continue

            if len(step_tuple) != 2:
                logger.error(f"Unexpected number of elements in step.step at step {index}: {len(step_tuple)}. Expected 2.")
                logger.debug(f"step.step Content: {step_tuple}")
                continue

            # Unpack the tuple
            player_states, dynamic_board_state = step_tuple

            # Construct the state dictionary
            state = {
                'player_states': player_states,
                'dynamic_board_state': dynamic_board_state
            }

            # Extract the action taken at this step
            action_taken = step.action  # This is an Action object

            # Reconstruct action_list as [player_color, action_type_str] + parameters
            # Assuming that action_taken.parameters is a list or tuple
            action_list = [action_taken.player_id.name, action_taken.action.name] + list(action_taken.parameters)

            # Validate action_list
            if not action_list or len(action_list) < 2:
                logger.warning(f"Skipping invalid action at step {index}. Action list: {action_list}")
                continue

            # Create Action instance from the action_list
            step_data = {
                'state': state,
                'action': action_list
            }

            action_obj = create_action_from_game_step(step_data, self.static_board_state)
            if action_obj is None:
                logger.warning(f"Skipping unsupported or invalid action at step {index}.")
                continue

            try:
                action_int = action_obj.flatten()
            except ValueError as e:
                logger.error(f"Error flattening action at step {index}: {e}")
                continue

            # Determine the current player from dynamic_board_state
            current_player = dynamic_board_state.current_player  # Assuming this is a PlayerID enum

            # Skip this turn if it's not the winning player's turn
            if self.game.winner != current_player:
                logger.debug(f"Skipping step {index} because it is not the winning player's turn.")
                continue

            # Get the player state of the winner
            winner_player_state = step.get_player_state_by_ID(self.game.winner)
            if winner_player_state is None:
                logger.warning(f"Winner player state not found at step {index}.")
                continue

            # Calculate reward
            reward = self.reward_function.calculate_reward(
                winner_player_state,
                action_list
            )

            # Create state tensors
            input_state_tensor = self.create_state_tensor(
                self.static_board_state,
                dynamic_board_state.to_dict(),  # Assuming DynamicBoardState can be converted to a dict
                [ps.to_dict() for ps in player_states]  # Assuming PlayerState can be converted to a dict
            )

            # Determine next state
            try:
                next_step = self.game.game_steps[index + 1]
                next_step_tuple = next_step.step

                # Unpack the next step tuple
                next_player_states, next_dynamic_board_state = next_step_tuple

                # Construct the next state dictionary
                next_state = {
                    'player_states': next_player_states,
                    'dynamic_board_state': next_dynamic_board_state
                }

                # Create next state tensor
                next_state_tensor = self.create_state_tensor(
                    self.static_board_state,
                    next_dynamic_board_state.to_dict(),
                    [ps.to_dict() for ps in next_player_states]
                )

                game_finished = 0
            except IndexError:
                # No next step; game is over
                logger.info(f"No next state prime found at step {index}! Game is over!")
                next_state_tensor = [-1] * INPUT_STATE_TENSOR_EXPECTED_LENGTH
                game_finished = 1

            yield input_state_tensor, action_int, reward, next_state_tensor, game_finished


def main():
    parser = CatanatronParser()

    # Replace the paths below with the correct paths to your board.json and data.json
    board_json_path = "datasets/2024-11-17_20_48_48/0a6c0574-3534-4f57-a802-a000ed8407ad/board.json"
    data_json_path = "datasets/2024-11-17_20_48_48/0a6c0574-3534-4f57-a802-a000ed8407ad/data.json"

    # Parse the static and dynamic board states
    logger.info("Parsing static board state...")
    static_board_state = parser.parse_board_json(board_json_path)
    logger.info("Static board state parsed successfully.")

    logger.info("Parsing game data...")
    catan_game = parser.parse_data_json(data_json_path, static_board_state)
    logger.info("Game data parsed successfully.")

    # Initialize the game iterator
    game_iterator = GameIterator(catan_game, static_board_state)

    # Iterate through the game and process each step
    for input_state, action_int, reward, next_state, game_finished in game_iterator.iterate_game():
        # Placeholder for processing the yielded data
        # For example, you might store it in a buffer for training
        # or directly pass it to a machine learning model
        logger.debug(f"Input State Tensor: {input_state}")
        logger.debug(f"Action Integer: {action_int}")
        logger.debug(f"Reward: {reward}")
        logger.debug(f"Next State Tensor: {next_state}")
        logger.debug(f"Game Finished: {game_finished}")

        # Example processing (to be replaced with actual logic)
        # process_training_data(input_state, action_int, reward, next_state, game_finished)

    logger.info("Game iteration completed.")


if __name__ == "__main__":
    main()
