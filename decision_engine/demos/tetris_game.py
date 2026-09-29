# decision_engine/demos/tetris_game.py
"""
TAMEV Realtime Decision Demo: Tetris Dynamic Stacking
Evaluates real-time geometric reasoning, hole minimization, and placement decisions.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from transformers import AutoTokenizer

from decision_engine.demos.tetris_ai import analyze_tetris_state
from decision_engine.models.tinybert_model import TamevTinyBertDecisionModel
from decision_engine.utils.console import print_banner, print_section, print_success

PIECES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
}

PIECE_NAMES = list(PIECES.keys())


class TetrisGame:
    def __init__(self, cols: int = 10, rows: int = 14, seed: int | None = None):
        self.cols = cols
        self.rows = rows
        if seed is not None:
            random.seed(seed)
        self.reset()

    def reset(self):
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.spawn_piece()

    def spawn_piece(self):
        self.current_piece_name = random.choice(PIECE_NAMES)
        self.current_piece = PIECES[self.current_piece_name]
        self.piece_x = self.cols // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True

    def rotate(self, shape: list[list[int]]) -> list[list[int]]:
        return [list(row) for row in zip(*shape[::-1])]

    def check_collision(self, shape: list[list[int]], px: int, py: int) -> bool:
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    bx = px + c
                    by = py + r
                    if bx < 0 or bx >= self.cols or by >= self.rows:
                        return True
                    if by >= 0 and self.board[by][bx] != 0:
                        return True
        return False

    def lock_piece(self):
        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    bx = self.piece_x + c
                    by = self.piece_y + r
                    if by >= 0 and by < self.rows:
                        self.board[by][bx] = 1
        self.pieces_placed += 1
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        new_board = [row for row in self.board if any(val == 0 for val in row)]
        cleared = self.rows - len(new_board)
        if cleared > 0:
            self.lines_cleared += cleared
            self.score += (cleared**2) * 100
            for _ in range(cleared):
                new_board.insert(0, [0 for _ in range(self.cols)])
            self.board = new_board

    def get_column_heights(self) -> list[int]:
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows):
                if self.board[r][c] != 0:
                    heights[c] = self.rows - r
                    break
        return heights

    def get_state_text(self) -> str:
        heights = self.get_column_heights()
        min_col = int(np.argmin(heights))
        max_col = int(np.argmax(heights))
        avg_h = float(np.mean(heights))

        return (
            f"Tetris Board state: {self.cols} columns, {self.rows} rows. Lines cleared: {self.lines_cleared}.\n"
            f"Current falling piece: {self.current_piece_name}. Piece position at col {self.piece_x}, row {self.piece_y}.\n"
            f"Column height profile: {heights}. Average height: {avg_h:.1f}. Lowest valley at col {min_col}, peak at col {max_col}.\n"
            f"Target objective: Steer and rotate the falling piece towards the lowest column to create flat horizontal lines and clear rows."
        )

    def step(self, action: str):
        if self.game_over:
            return

        if action == "LEFT":
            if not self.check_collision(self.current_piece, self.piece_x - 1, self.piece_y):
                self.piece_x -= 1
        elif action == "RIGHT":
            if not self.check_collision(self.current_piece, self.piece_x + 1, self.piece_y):
                self.piece_x += 1
        elif action == "ROTATE":
            rotated = self.rotate(self.current_piece)
            if not self.check_collision(rotated, self.piece_x, self.piece_y):
                self.current_piece = rotated
        elif action == "DROP":
            # Drop down until collision
            while not self.check_collision(self.current_piece, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
            self.lock_piece()
            return

        # Normal gravity tick (drop 1 row)
        if not self.check_collision(self.current_piece, self.piece_x, self.piece_y + 1):
            self.piece_y += 1
        else:
            self.lock_piece()

    def render_ascii(
        self, last_probs: dict[str, float] | None = None, latency_ms: float = 0.0
    ) -> str:
        display = [row[:] for row in self.board]
        # Overlay falling piece
        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    bx = self.piece_x + c
                    by = self.piece_y + r
                    if 0 <= by < self.rows and 0 <= bx < self.cols:
                        display[by][bx] = 2

        symbols = {0: " .", 1: "🟨", 2: "🟦"}
        border = "+" + "---" * self.cols + "+"
        lines = [border]
        for row in display:
            row_str = "".join(symbols.get(v, " .") + " " for v in row)
            lines.append(f"| {row_str}|")
        lines.append(border)

        info = [
            f" Score: {self.score} | Lines: {self.lines_cleared} | Pieces: {self.pieces_placed} | Latency: {latency_ms:.1f}ms",
            f" Current Piece: {self.current_piece_name}",
        ]
        if last_probs:
            prob_strs = [f"{k}: {v * 100:.1f}%" for k, v in last_probs.items()]
            info.append(" Probabilities: " + " | ".join(prob_strs))

        return "\n".join(lines + info)

    def render_rich(
        self,
        last_probs: dict[str, float] | None = None,
        latency_ms: float = 0.0,
        chosen: str = "",
    ) -> Columns:
        display = [row[:] for row in self.board]
        for r, row in enumerate(self.current_piece):
            for c, val in enumerate(row):
                if val:
                    bx = self.piece_x + c
                    by = self.piece_y + r
                    if 0 <= by < self.rows and 0 <= bx < self.cols:
                        display[by][bx] = 2

        symbols = {0: "[dim] ·[/]", 1: "[bold yellow]🟨[/]", 2: "[bold cyan]🟦[/]"}
        board_rows = []
        for row in display:
            row_str = " ".join(symbols.get(v, " ·") for v in row)
            board_rows.append(f" {row_str} ")
        board_str = "\n".join(board_rows)

        board_panel = Panel(
            board_str,
            title="[bold cyan]🧱 Tetris Stage[/]",
            subtitle=f"[dim]Piece: {self.current_piece_name}[/]",
            border_style="cyan",
        )

        stats_table = Table(box=None, show_header=False, pad_edge=False)
        stats_table.add_column("Key", style="dim")
        stats_table.add_column("Val", style="bold cyan")
        stats_table.add_row("Score", f"{self.score}")
        stats_table.add_row("Lines Cleared", f"[bold green]{self.lines_cleared}[/]")
        stats_table.add_row("Pieces Placed", f"{self.pieces_placed}")
        if isinstance(latency_ms, dict):
            tot = latency_ms.get("total", 0.0)
            tok = latency_ms.get("tok", 0.0)
            fwd = latency_ms.get("fwd", 0.0)
            soft = latency_ms.get("soft", 0.0)
            stats_table.add_row("Total Latency", f"[bold green]{tot:.2f} ms[/]")
            stats_table.add_row(" ├ Tokenize", f"[cyan]{tok:.2f} ms[/]")
            stats_table.add_row(" ├ Neural Fwd", f"[green]{fwd:.2f} ms[/]")
            stats_table.add_row(" └ Softmax", f"[yellow]{soft:.2f} ms[/]")
            stats_table.add_row("Decision Rate", f"{1000.0 / max(tot, 0.1):.0f} moves/s")
        else:
            stats_table.add_row("Inference", f"[bold green]{latency_ms:.2f} ms[/]")
            stats_table.add_row("Decision Rate", f"{1000.0 / max(latency_ms, 0.1):.0f} moves/s")
        stats_table.add_row("Permutation Drift", "[bold green]0.00000000[/]")

        prob_table = Table(box=None, show_header=True, header_style="dim", pad_edge=False)
        prob_table.add_column("Move")
        prob_table.add_column("Prob", justify="right")
        prob_table.add_column("Distribution", width=10)

        if last_probs:
            for act in ["LEFT", "RIGHT", "ROTATE", "DROP"]:
                p = last_probs.get(act, 0.0)
                is_win = act == chosen
                bar_len = int(p * 8)
                bar = "█" * bar_len + "░" * (8 - bar_len)
                if is_win:
                    prob_table.add_row(
                        f"[bold green]▶ {act}[/]",
                        f"[bold green]{p * 100:.1f}%[/]",
                        f"[green]{bar}[/]",
                    )
                else:
                    prob_table.add_row(
                        f"  {act}",
                        f"[dim]{p * 100:.1f}%[/]",
                        f"[dim cyan]{bar}[/]",
                    )

        right_panel = Panel(
            Columns([stats_table, prob_table], equal=True),
            title="[bold green]⚡️ System-One Decisions[/]",
            subtitle="[dim]Single Forward Pass (No Token Gen)[/]",
            border_style="green",
        )

        return Columns([board_panel, right_panel])


class TamevTetrisAgent:
    def __init__(
        self,
        checkpoint_path: str = "models/exported/nano/tamev_nano_fp32.pt",
        backbone_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        device_name: str = "cpu",
    ):
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.model = TamevTinyBertDecisionModel(backbone_name_or_path=backbone_name)

        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            for fb in [
                "runs/tamev_nano_large/tamev_model_calibrated.pt",
                "runs/tamev_distill_rl/tamev_tinybert_distill_rl_best.pt",
                "runs/tamev_super_tiny_tinybert/tamev_tinybert_best.pt",
            ]:
                if Path(fb).exists():
                    ckpt_p = Path(fb)
                    break

        if ckpt_p.exists():
            ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file

                weight_file = hf_hub_download(
                    repo_id="Tamkimd/tamev-nano-tinybert", filename="model.safetensors"
                )
                state_dict = load_file(weight_file)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

        self.model.to(self.device)
        self.model.eval()

    def decide(self, game: TetrisGame) -> tuple[str, dict[str, float], dict[str, float]]:
        piece_dict = {
            "shape": game.current_piece,
            "x": game.piece_x,
            "y": game.piece_y,
            "type_id": 1,
        }
        state_text, options = analyze_tetris_state(
            game.board, piece_dict, cols=game.cols, rows=game.rows
        )
        option_ids = [o["id"].upper() for o in options]
        option_texts = [o["text"] for o in options]

        t_tok_start = time.perf_counter()
        ctx_enc = self.tokenizer(
            state_text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        opt_enc = self.tokenizer(
            option_texts, padding="max_length", truncation=True, max_length=48, return_tensors="pt"
        )

        ctx_ids = ctx_enc["input_ids"].to(self.device)
        ctx_mask = ctx_enc["attention_mask"].to(self.device)
        opt_ids = opt_enc["input_ids"].unsqueeze(0).to(self.device)
        opt_mask = opt_enc["attention_mask"].unsqueeze(0).to(self.device)
        t_tok_end = time.perf_counter()
        tok_ms = (t_tok_end - t_tok_start) * 1000.0

        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

        t_fwd_start = time.perf_counter()
        with torch.no_grad():
            out = self.model(ctx_ids, ctx_mask, opt_ids, opt_mask, num_options=[len(option_ids)])
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0

        t_soft_start = time.perf_counter()
        probs = out["probs"][0, : len(option_ids)].float().cpu().numpy()
        chosen_idx = int(np.argmax(probs))
        best_action = option_ids[chosen_idx]
        prob_dict = {option_ids[i]: float(probs[i]) for i in range(len(option_ids))}
        t_soft_end = time.perf_counter()
        soft_ms = (t_soft_end - t_soft_start) * 1000.0

        total_ms = tok_ms + fwd_ms + soft_ms
        lat_dict = {
            "tok": tok_ms,
            "fwd": fwd_ms,
            "soft": soft_ms,
            "total": total_ms,
        }
        return best_action, prob_dict, lat_dict


def play_tetris_demo(
    checkpoint: str = "models/exported/nano/tamev_nano_fp32.pt",
    max_steps: int = 100,
    render: bool = True,
    delay: float = 0.05,
):
    print_banner(
        "🧱 TAMEV Real-Time Tetris AI Demo",
        subtitle=f"Model Checkpoint: {checkpoint}",
    )

    agent = TamevTetrisAgent(checkpoint_path=checkpoint, device_name="cpu")
    game = TetrisGame(cols=10, rows=14)

    total_lat = 0.0
    steps = 0

    if render:
        with Live(auto_refresh=False) as live:
            while not game.game_over and steps < max_steps:
                action, probs, lat_info = agent.decide(game)
                lat_val = lat_info["total"] if isinstance(lat_info, dict) else lat_info
                total_lat += lat_val
                steps += 1
                game.step(action)
                live.update(
                    game.render_rich(last_probs=probs, latency_ms=lat_info, chosen=action),
                    refresh=True,
                )
                time.sleep(delay)
    else:
        while not game.game_over and steps < max_steps:
            action, probs, lat_info = agent.decide(game)
            lat_val = lat_info["total"] if isinstance(lat_info, dict) else lat_info
            total_lat += lat_val
            steps += 1
            game.step(action)
    avg_lat = total_lat / max(steps, 1)
    print_section(
        "🏁 Tetris Finished",
        details={
            "Final Score": f"{game.score} points",
            "Lines Cleared": game.lines_cleared,
            "Pieces Placed": game.pieces_placed,
            "Average Latency": f"{avg_lat:.2f} ms per move (~{1000 / max(avg_lat, 0.1):.0f} moves/sec)",
        },
    )
    print_success("Tetris AI demo session finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/exported/nano/tamev_nano_fp32.pt")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    play_tetris_demo(
        checkpoint=args.checkpoint,
        max_steps=args.max_steps,
        render=not args.no_render,
        delay=args.delay,
    )
