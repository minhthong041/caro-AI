import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.minimax import MinimaxAI
from src.board import Board
from src.config import BOARD_SIZE


@dataclass
class GameResult:
    winner: int
    total_moves: int
    thinking_time: dict
    invalid_moves: dict


class RandomAgent:
    name = "random"

    def get_best_move(self, grid, player=None):
        valid_moves = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if grid[r][c] == 0
        ]
        return random.choice(valid_moves) if valid_moves else None


class MinimaxAgent:
    def __init__(self, player, depth):
        opponent = 2 if player == 1 else 1
        self.name = f"minimax:{depth}"
        self.ai = MinimaxAI(ai_player=player, human_player=opponent, depth=depth)

    def get_best_move(self, grid, player=None):
        return self.ai.get_best_move(grid)


class CnnAgent:
    name = "cnn"

    def __init__(self):
        from ai.supervised import SupervisedAI

        self.ai = SupervisedAI()
        if self.ai.model is None:
            raise RuntimeError("CNN model not found")

    def get_best_move(self, grid, player=None):
        return self.ai.get_best_move(grid, player=player)


def parse_agent_spec(spec):
    normalized = spec.strip().lower()
    if normalized == "random":
        return ("random", None)
    if normalized == "cnn":
        return ("cnn", None)
    if normalized.startswith("minimax"):
        parts = normalized.split(":", 1)
        depth = 3 if len(parts) == 1 else int(parts[1])
        if depth < 1:
            raise ValueError("minimax depth must be >= 1")
        return ("minimax", depth)
    raise ValueError(f"unknown agent spec: {spec}")


def build_agent(spec, player):
    kind, value = parse_agent_spec(spec)
    if kind == "random":
        return RandomAgent()
    if kind == "cnn":
        return CnnAgent()
    return MinimaxAgent(player=player, depth=value)


def normalize_move(grid, move):
    if not isinstance(move, (list, tuple)) or len(move) != 2:
        return None
    try:
        row, col = int(move[0]), int(move[1])
    except (TypeError, ValueError):
        return None
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and grid[row][col] == 0:
        return (row, col)
    return None


def get_fallback_move(grid):
    valid_moves = [
        (r, c)
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if grid[r][c] == 0
    ]
    return random.choice(valid_moves) if valid_moves else None


def simulate_game(p1_spec, p2_spec, max_moves=BOARD_SIZE * BOARD_SIZE):
    board = Board()
    agents = {
        1: build_agent(p1_spec, player=1),
        2: build_agent(p2_spec, player=2),
    }
    thinking_time = {1: 0.0, 2: 0.0}
    invalid_moves = {1: 0, 2: 0}
    current_player = 1

    for move_count in range(1, max_moves + 1):
        grid_copy = [row[:] for row in board.grid]
        started_at = time.perf_counter()
        move = agents[current_player].get_best_move(grid_copy, player=current_player)
        thinking_time[current_player] += time.perf_counter() - started_at

        move = normalize_move(board.grid, move)
        if move is None:
            invalid_moves[current_player] += 1
            move = get_fallback_move(board.grid)
        if move is None:
            return GameResult(
                winner=0,
                total_moves=move_count - 1,
                thinking_time=thinking_time,
                invalid_moves=invalid_moves,
            )

        row, col = move
        board.place_piece(row, col, current_player)
        if board.check_win(row, col, current_player):
            return GameResult(
                winner=current_player,
                total_moves=move_count,
                thinking_time=thinking_time,
                invalid_moves=invalid_moves,
            )
        current_player = 2 if current_player == 1 else 1

    return GameResult(
        winner=0,
        total_moves=max_moves,
        thinking_time=thinking_time,
        invalid_moves=invalid_moves,
    )


def run_matchup(p1_spec, p2_spec, games, max_moves):
    summary = {
        "p1": p1_spec,
        "p2": p2_spec,
        "games": games,
        "p1_wins": 0,
        "p2_wins": 0,
        "draws": 0,
        "avg_moves": 0.0,
        "p1_avg_time": 0.0,
        "p2_avg_time": 0.0,
        "p1_invalid": 0,
        "p2_invalid": 0,
    }

    for _ in range(games):
        result = simulate_game(p1_spec, p2_spec, max_moves=max_moves)
        if result.winner == 1:
            summary["p1_wins"] += 1
        elif result.winner == 2:
            summary["p2_wins"] += 1
        else:
            summary["draws"] += 1
        summary["avg_moves"] += result.total_moves
        summary["p1_avg_time"] += result.thinking_time[1]
        summary["p2_avg_time"] += result.thinking_time[2]
        summary["p1_invalid"] += result.invalid_moves[1]
        summary["p2_invalid"] += result.invalid_moves[2]

    summary["avg_moves"] /= games
    summary["p1_avg_time"] /= games
    summary["p2_avg_time"] /= games
    return summary


def print_table(rows):
    headers = [
        "P1",
        "P2",
        "Games",
        "P1W",
        "P2W",
        "Draw",
        "AvgMoves",
        "P1Time",
        "P2Time",
        "Invalid",
    ]
    lines = []
    for row in rows:
        lines.append([
            row["p1"],
            row["p2"],
            str(row["games"]),
            str(row["p1_wins"]),
            str(row["p2_wins"]),
            str(row["draws"]),
            f"{row['avg_moves']:.1f}",
            f"{row['p1_avg_time']:.3f}s",
            f"{row['p2_avg_time']:.3f}s",
            f"{row['p1_invalid']}/{row['p2_invalid']}",
        ])

    widths = [
        max(len(headers[i]), *(len(line[i]) for line in lines))
        for i in range(len(headers))
    ]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for line in lines:
        print(" | ".join(line[i].ljust(widths[i]) for i in range(len(headers))))


def build_default_matchups(include_cnn=False, include_depth_suite=False):
    matchups = [
        ("minimax:3", "random"),
        ("random", "minimax:3"),
    ]
    if include_depth_suite:
        matchups.extend([
            ("minimax:3", "minimax:2"),
            ("minimax:2", "minimax:3"),
        ])
    if include_cnn:
        matchups.extend([
            ("cnn", "minimax:1"),
            ("minimax:1", "cnn"),
        ])
    return matchups


def main():
    parser = argparse.ArgumentParser(description="Benchmark Caro AI agents without opening Pygame.")
    parser.add_argument("--games", type=int, default=6, help="games per matchup")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max-moves", type=int, default=BOARD_SIZE * BOARD_SIZE)
    parser.add_argument("--p1", help="agent spec for player 1, e.g. random, minimax:3, cnn")
    parser.add_argument("--p2", help="agent spec for player 2, e.g. random, minimax:3, cnn")
    parser.add_argument("--include-depth-suite", action="store_true", help="include slower minimax depth-vs-depth matchups")
    parser.add_argument("--include-cnn", action="store_true", help="include CNN in the default benchmark suite")
    args = parser.parse_args()

    if args.games < 1:
        raise SystemExit("--games must be >= 1")
    if (args.p1 is None) != (args.p2 is None):
        raise SystemExit("--p1 and --p2 must be provided together")

    random.seed(args.seed)
    matchups = (
        [(args.p1, args.p2)]
        if args.p1
        else build_default_matchups(args.include_cnn, args.include_depth_suite)
    )
    rows = []
    for p1_spec, p2_spec in matchups:
        try:
            rows.append(run_matchup(p1_spec, p2_spec, args.games, args.max_moves))
        except Exception as exc:
            print(f"Skipped {p1_spec} vs {p2_spec}: {exc}")

    if rows:
        print_table(rows)


if __name__ == "__main__":
    main()
