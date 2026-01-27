#!/usr/bin/env python3
"""
Stockfish Evaluation Script for ELO Estimation

Plays the trained model against Stockfish at various skill levels
to estimate its ELO rating.

CLI Options:                                                                                                                                                                                                    
- -l/--skill-levels - Skill levels to test (default: 0 3 5 8 10)                                                                         
- -g/--games - Games per level (default: 20)                                                                                             
- -n/--simulations - MCTS simulations per move (default: 800)                                                                            
- -t/--stockfish-time - Stockfish time limit (default: 1.0s)                                                                             
- -v/--verbose - Print individual moves 
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine
import torch

from mcts import MCTS
from nn import ChessNet
from utils import ActionMapper


# Approximate ELO mapping for Stockfish skill levels
# Based on community estimates; actual values vary by Stockfish version
STOCKFISH_PATH = "./stockfish"
MODEL_PATH = "rl_chess_model_latest.pth"

SKILL_LEVEL_ELO = {
    0: 800,
    1: 1000,
    2: 1200,
    3: 1400,
    4: 1700,
    5: 2000,
    6: 2300,
    7: 2700,
    8: 3000
}


@dataclass
class GameResult:
    """Result of a single game."""
    winner: Optional[chess.Color]  # None for draw
    model_color: chess.Color
    moves: int
    termination: str

    @property
    def model_won(self) -> bool:
        return self.winner == self.model_color

    @property
    def model_lost(self) -> bool:
        return self.winner is not None and self.winner != self.model_color

    @property
    def is_draw(self) -> bool:
        return self.winner is None


class EloEstimator:
    """Estimates ELO rating from game results against known opponents."""

    @staticmethod
    def calculate_score(wins: int, draws: int, total: int) -> float:
        """Calculate score as wins + 0.5*draws / total."""
        if total == 0:
            return 0.5
        return (wins + 0.5 * draws) / total

    @staticmethod
    def estimate_elo(score: float, opponent_elo: int) -> Optional[float]:
        """
        Estimate ELO from win rate against a known opponent.

        Uses the formula: ELO = opponent_elo + 400 * log10(score / (1 - score))

        Returns None if score is 0 or 1 (undefined).
        """
        if score <= 0 or score >= 1:
            return None

        elo_diff = 400 * math.log10(score / (1 - score))
        return opponent_elo + elo_diff


class StockfishEvaluator:
    """Evaluates the trained model against Stockfish."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        simulations: int = 800,
        stockfish_time: float = 1.0,
        verbose: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the trained model weights.
            simulations: Number of MCTS simulations per move.
            stockfish_time: Time limit for Stockfish moves (seconds).
            verbose: Print individual moves.
        """
        self.stockfish_path = STOCKFISH_PATH
        self.model_path = MODEL_PATH
        self.simulations = simulations
        self.stockfish_time = stockfish_time
        self.verbose = verbose

        # Load model
        self.device = self._get_device()
        self.mapper = ActionMapper()
        self.model = self._load_model()
        self.mcts = MCTS(self.model, self.device, self.mapper)

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> ChessNet:
        """Load the trained model."""
        model = ChessNet(action_size=self.mapper.vocab_size).to(self.device)
        model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        model.eval()
        return model

    def get_model_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get the best move from MCTS."""
        root = self.mcts.search(board, num_simulations=self.simulations)

        best_move = None
        max_visits = -1
        for move, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_move = move

        return best_move

    def play_game(
        self,
        engine: chess.engine.SimpleEngine,
        skill_level: int,
        model_plays_white: bool,
    ) -> GameResult:
        """
        Play a single game between the model and Stockfish.

        Args:
            engine: Stockfish engine instance.
            skill_level: Stockfish skill level (0-8).
            model_plays_white: Whether the model plays as white.

        Returns:
            GameResult with game outcome details.
        """
        engine.configure({"Skill Level": skill_level})
        board = chess.Board()
        model_color = chess.WHITE if model_plays_white else chess.BLACK

        move_count = 0
        while not board.is_game_over():
            is_model_turn = board.turn == model_color

            if is_model_turn:
                move = self.get_model_move(board)
                if move is None:
                    break
            else:
                result = engine.play(
                    board, chess.engine.Limit(time=self.stockfish_time)
                )
                move = result.move

            if self.verbose:
                player = "Model" if is_model_turn else f"Stockfish(L{skill_level})"
                print(f"  {move_count + 1}. {player}: {move}")

            board.push(move)
            move_count += 1

        # Determine winner
        outcome = board.outcome()
        winner = outcome.winner if outcome else None
        termination = outcome.termination.name if outcome else "UNKNOWN"

        return GameResult(
            winner=winner,
            model_color=model_color,
            moves=move_count,
            termination=termination,
        )

    def run_evaluation(
        self, skill_levels: list[int], games_per_level: int
    ) -> dict[int, dict]:
        """
        Run evaluation across multiple skill levels.

        Args:
            skill_levels: List of Stockfish skill levels to test.
            games_per_level: Number of games per skill level (half as white, half as black).

        Returns:
            Dictionary mapping skill level to results.
        """
        results = {}

        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as e:
            print(f"Error: Could not start Stockfish at '{self.stockfish_path}'")
            print(f"  {e}")
            sys.exit(1)

        try:
            for skill_level in skill_levels:
                print(f"\n{'='*60}")
                print(f"Testing against Stockfish Skill Level {skill_level}")
                print(f"  Estimated ELO: ~{SKILL_LEVEL_ELO.get(skill_level, 'Unknown')}")
                print(f"{'='*60}")

                wins = 0
                draws = 0
                losses = 0
                total_moves = 0

                # Play half games as white, half as black
                games_as_white = games_per_level // 2

                for game_num in range(games_per_level):
                    model_plays_white = game_num < games_as_white
                    color_str = "White" if model_plays_white else "Black"

                    print(f"\nGame {game_num + 1}/{games_per_level} (Model as {color_str})...")

                    result = self.play_game(engine, skill_level, model_plays_white)
                    total_moves += result.moves

                    if result.model_won:
                        wins += 1
                        outcome_str = "WIN"
                    elif result.model_lost:
                        losses += 1
                        outcome_str = "LOSS"
                    else:
                        draws += 1
                        outcome_str = "DRAW"

                    print(
                        f"  Result: {outcome_str} "
                        f"({result.termination}, {result.moves} moves)"
                    )
                    print(f"  Running: W{wins}-D{draws}-L{losses}")

                # Calculate statistics for this skill level
                total = wins + draws + losses
                score = EloEstimator.calculate_score(wins, draws, total)
                opponent_elo = SKILL_LEVEL_ELO[skill_level]
                estimated_elo = EloEstimator.estimate_elo(score, opponent_elo)

                results[skill_level] = {
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "score": score,
                    "avg_moves": total_moves / total if total > 0 else 0,
                    "opponent_elo": opponent_elo,
                    "estimated_elo": estimated_elo,
                }

                print(f"\nSkill Level {skill_level} Summary:")
                print(f"  Record: {wins}W - {draws}D - {losses}L")
                print(f"  Score: {score:.1%}")
                if estimated_elo:
                    print(f"  Estimated ELO vs this level: {estimated_elo:.0f}")
                else:
                    print(f"  Estimated ELO: N/A (score too extreme)")

        finally:
            engine.quit()

        return results


def print_final_summary(results: dict[int, dict]) -> None:
    """Print a final summary of all results."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    total_wins = 0
    total_draws = 0
    total_losses = 0
    elo_estimates = []

    print(f"\n{'Level':<8} {'Opp ELO':<10} {'Record':<12} {'Score':<10} {'Est. ELO':<10}")
    print("-" * 50)

    for level in sorted(results.keys()):
        r = results[level]
        total_wins += r["wins"]
        total_draws += r["draws"]
        total_losses += r["losses"]

        record = f"{r['wins']}W-{r['draws']}D-{r['losses']}L"
        elo_str = f"{r['estimated_elo']:.0f}" if r["estimated_elo"] else "N/A"

        if r["estimated_elo"]:
            elo_estimates.append(r["estimated_elo"])

        print(
            f"{level:<8} {r['opponent_elo']:<10} {record:<12} "
            f"{r['score']:.1%}     {elo_str:<10}"
        )

    print("-" * 50)
    total_games = total_wins + total_draws + total_losses
    overall_score = EloEstimator.calculate_score(total_wins, total_draws, total_games)
    print(
        f"{'Total':<8} {'':<10} "
        f"{total_wins}W-{total_draws}D-{total_losses}L   {overall_score:.1%}"
    )

    if elo_estimates:
        avg_elo = sum(elo_estimates) / len(elo_estimates)
        print(f"\nAverage Estimated ELO: {avg_elo:.0f}")
        print(f"ELO Range: {min(elo_estimates):.0f} - {max(elo_estimates):.0f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained chess model against Stockfish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_stockfish.py
  python evaluate_stockfish.py -l 0 5 10 -g 20
  python evaluate_stockfish.py -n 800 -v
        """,
    )

    parser.add_argument(
        "--skill-levels", "-l",
        type=int,
        nargs="+",
        default=[0, 3, 5, 8, 10],
        help="Stockfish skill levels to test (default: 0 3 5 8 10)",
    )
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=20,
        help="Games per skill level (default: 20)",
    )
    parser.add_argument(
        "--simulations", "-n",
        type=int,
        default=800,
        help="MCTS simulations per move (default: 800)",
    )
    parser.add_argument(
        "--stockfish-time", "-t",
        type=float,
        default=1.0,
        help="Time limit for Stockfish moves in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print individual moves",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Validate skill levels
    for level in args.skill_levels:
        if not 0 <= level <= 20:
            print(f"Error: Skill level {level} must be between 0 and 20")
            sys.exit(1)

    print("=" * 60)
    print("STOCKFISH EVALUATION")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Skill levels: {args.skill_levels}")
    print(f"Games per level: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Stockfish time: {args.stockfish_time}s")

    evaluator = StockfishEvaluator(
        model_path=MODEL_PATH,
        simulations=args.simulations,
        stockfish_time=args.stockfish_time,
        verbose=args.verbose,
    )

    print(f"\nUsing device: {evaluator.device}")

    results = evaluator.run_evaluation(
        skill_levels=args.skill_levels,
        games_per_level=args.games,
    )

    print_final_summary(results)


if __name__ == "__main__":
    main()
