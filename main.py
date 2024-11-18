#!/usr/bin/env python3

import os
import argparse
from typing import Iterator, Tuple, List

from interfaces.catanatron_interface import CatanatronParser
from environment.board_state import StaticBoardState
from environment.game import CatanGame

def process_directory_iterator(base_dir: str) -> Iterator[Tuple[str, str]]:
    for root, dirs, _ in os.walk(base_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            board_file = os.path.join(subdir_path, "board.json")
            data_file = os.path.join(subdir_path, "data.json")

            if os.path.exists(board_file) and os.path.exists(data_file):
                yield board_file, data_file

def parse_data(board_path, data_path) -> Tuple[StaticBoardState, CatanGame]:
    parser = CatanatronParser()
    static_board_state = parser.parse_board_json(board_path)
    game = parser.parse_data_json(data_path, static_board_state)
    return [static_board_state, game]

def main():

    parser = argparse.ArgumentParser(description="Parse Catan board.json and data.json files in subdirectories within a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the base directory containing training dataset.")
    args = parser.parse_args()

    # Process the directory
    if os.path.isdir(args.dataset_dir):
        for board_path, data_path in process_directory_iterator(args.dataset_dir):
            print(f"Processing: {board_path}, {data_path}")
            static_board_state, game = parse_data(board_path, data_path)
            print(static_board_state)
            return
    else:
        print(f"The specified path {args.dataset_dir} is not a directory.")


if __name__ == "__main__":
    main()
