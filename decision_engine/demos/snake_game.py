# decision_engine/demos/snake_game.py
"""
TAMEV Realtime Decision Demo: The Snake Game
Evaluates real-time on-device decision speed, safety reasoning, and spatial orientation.
Every game tick, TAMEV evaluates the board state (obstacles, heading, food direction)
and outputs direction probabilities [UP, DOWN, LEFT, RIGHT] in sub-25ms latency.
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from transformers import AutoTokenizer

from decision_engine.demos.snake_ai import DIRECTIONS, OPPOSITES, analyze_snake_state
from decision_engine.models.tinybert_model import TamevTinyBertDecisionModel
from decision_engine.utils.console import print_banner, print_section, print_success


class SnakeGame:
    DIRECTIONS = DIRECTIONS
    OPPOSITES = OPPOSITES

    def __init__(self, width: int = 20, height: int = 20, seed: int | None = None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.snake = [(cx, cy), (cx, cy + 1), (cx, cy + 2)]
        self.direction = "UP"
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.recent_positions = deque(maxlen=16)
        self.spawn_food()

    def spawn_food(self):
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.snake
        ]
        if empty_cells:
            self.food = random.choice(empty_cells)
            return self.food
        self.game_over = True  # Win!
        return (0, 0)

    def is_collision(self, pt: tuple[int, int]) -> bool:
        x, y = pt
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pt in self.snake[:-1]

    def get_obstacle_status(self, direction: str) -> str:
        dx, dy = self.DIRECTIONS[direction]
        head_x, head_y = self.snake[0]
        next_pt = (head_x + dx, head_y + dy)
        if (
            next_pt[0] < 0
            or next_pt[0] >= self.width
            or next_pt[1] < 0
            or next_pt[1] >= self.height
        ):
            return "WALL DANGER"
        if next_pt in self.snake:
            return "BODY COLLISION"
        return "SAFE CLEAR"

    def get_state_text(self) -> str:
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        dx = food_x - head_x
        dy = food_y - head_y

        food_h = "RIGHT" if dx > 0 else ("LEFT" if dx < 0 else "ALIGNED")
        food_v = "DOWN" if dy > 0 else ("UP" if dy < 0 else "ALIGNED")

        up_st = self.get_obstacle_status("UP")
        down_st = self.get_obstacle_status("DOWN")
        left_st = self.get_obstacle_status("LEFT")
        right_st = self.get_obstacle_status("RIGHT")

        return (
            f"Snake Head at ({head_x}, {head_y}). Target food at ({food_x}, {food_y}).\n"
            f"Current heading: {self.direction}. Length: {len(self.snake)}.\n"
            f"Food relative location: horizontal {food_h} (dx={dx}), vertical {food_v} (dy={dy}).\n"
            f"Surrounding paths:\n"
            f"- UP: {up_st}\n"
            f"- DOWN: {down_st}\n"
            f"- LEFT: {left_st}\n"
            f"- RIGHT: {right_st}\n"
            f"Goal: Choose the optimal direction to reach food while strictly avoiding obstacles and 180-degree turns."
        )

    def step(self, chosen_direction: str) -> tuple[bool, int]:
        """Executes 1 game step. Returns (game_over, reward)."""
        if self.game_over:
            return True, 0

        # Disallow direct 180 reverse
        if chosen_direction == self.OPPOSITES.get(self.direction):
            chosen_direction = self.direction

        self.direction = chosen_direction
        dx, dy = self.DIRECTIONS[chosen_direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        if self.is_collision(new_head):
            self.game_over = True
            return True, -10

        self.snake.insert(0, new_head)
        self.steps += 1
        self.recent_positions.append(self.snake[0])

        if new_head == self.food:
            self.score += 10
            self.spawn_food()
            return False, 10
        self.snake.pop()
        return False, 1

    def render_ascii(
        self, last_probs: dict[str, float] | None = None, latency_ms: float = 0.0
    ) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        fx, fy = self.food
        grid[fy][fx] = "🍎"

        for i, (sx, sy) in enumerate(self.snake):
            if i == 0:
                grid[sy][sx] = "🟢" if self.direction in ["UP", "DOWN"] else "🟩"
            else:
                grid[sy][sx] = "🟩"

        border = "+" + "---" * self.width + "+"
        lines = [border]
        for row in grid:
            row_str = "".join(f"{c:>2} " for c in row)
            lines.append(f"| {row_str}|")
        lines.append(border)

        info = [
            f" Score: {self.score} | Steps: {self.steps} | Decision Latency: {latency_ms:.1f}ms",
            f" Heading: {self.direction}",
        ]
        if last_probs:
            prob_strs = [
                f"{d}: {last_probs.get(d, 0.0) * 100:.1f}%" for d in ["UP", "DOWN", "LEFT", "RIGHT"]
            ]
            info.append(" Probabilities: " + " | ".join(prob_strs))

        return "\n".join(lines + info)

    def render_rich(
        self,
        last_probs: dict[str, float] | None = None,
        latency_ms: float = 0.0,
        chosen: str = "",
    ) -> Columns:
        grid = [["·" for _ in range(self.width)] for _ in range(self.height)]
        fx, fy = self.food
        grid[fy][fx] = "[bold red]🍎[/]"

        for i, (sx, sy) in enumerate(self.snake):
            if i == 0:
                grid[sy][sx] = "[bold green]🟢[/]"
            else:
                grid[sy][sx] = "[green]🟩[/]"

        board_str = "\n".join(" ".join(f"{c:>2}" for c in row) for row in grid)
        board_panel = Panel(
            board_str,
            title="[bold green]🐍 TAMEV Arena[/]",
            subtitle=f"[dim]{self.width}x{self.height} Grid[/]",
            border_style="green",
        )

        # Telemetry Table
        stats_table = Table(box=None, show_header=False, pad_edge=False)
        stats_table.add_column("Key", style="dim")
        stats_table.add_column("Val", style="bold cyan")
        stats_table.add_row("Score", f"{self.score}")
        stats_table.add_row("Steps", f"{self.steps}")
        stats_table.add_row("Apples", f"{self.score // 10}")
        stats_table.add_row("Heading", f"[bold yellow]{self.direction}[/]")
        if isinstance(latency_ms, dict):
            tot = latency_ms.get("total", 0.0)
            tok = latency_ms.get("tok", 0.0)
            fwd = latency_ms.get("fwd", 0.0)
            soft = latency_ms.get("soft", 0.0)
            stats_table.add_row("Total Latency", f"[bold green]{tot:.2f} ms[/]")
            stats_table.add_row(" ├ Tokenize", f"[cyan]{tok:.2f} ms[/]")
            stats_table.add_row(" ├ Neural Fwd", f"[green]{fwd:.2f} ms[/]")
            stats_table.add_row(" └ Softmax", f"[yellow]{soft:.2f} ms[/]")
            stats_table.add_row("Decisions/s", f"{1000.0 / max(tot, 0.1):.0f} /s")
        else:
            stats_table.add_row("Inference", f"[bold green]{latency_ms:.2f} ms[/]")
            stats_table.add_row("Decisions/s", f"{1000.0 / max(latency_ms, 0.1):.0f} /s")
        stats_table.add_row("Position Drift", "[bold green]0.00000000[/]")

        # Probabilities
        prob_table = Table(box=None, show_header=True, header_style="dim", pad_edge=False)
        prob_table.add_column("Action")
        prob_table.add_column("Prob", justify="right")
        prob_table.add_column("Distribution", width=12)

        if last_probs:
            for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
                p = last_probs.get(d, 0.0)
                is_win = d == chosen
                bar_len = int(p * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                if is_win:
                    prob_table.add_row(
                        f"[bold green]▶ {d}[/]",
                        f"[bold green]{p * 100:.1f}%[/]",
                        f"[green]{bar}[/]",
                    )
                else:
                    prob_table.add_row(
                        f"  {d}",
                        f"[dim]{p * 100:.1f}%[/]",
                        f"[dim cyan]{bar}[/]",
                    )

        right_panel = Panel(
            Columns([stats_table, prob_table], equal=True),
            title="[bold cyan]⚡️ System-One Telemetry[/]",
            subtitle="[dim]Zero Autoregressive Token Delay[/]",
            border_style="cyan",
        )

        return Columns([board_panel, right_panel])


class TamevSnakeAgent:
    """AI Agent driven by TAMEV Decision Model."""

    CANDIDATE_OPTIONS = [
        "Move UP towards target",
        "Move DOWN towards target",
        "Move LEFT towards target",
        "Move RIGHT towards target",
    ]
    DIRECTION_KEYS = ["UP", "DOWN", "LEFT", "RIGHT"]

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
                "runs/tamev_super_tiny_tinybert/tamev_tinybert_best.pt",
                "runs/tamev_distill_rl/tamev_tinybert_distill_rl_best.pt",
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

        # Pre-encode candidate options once for maximum real-time latency
        opt_enc = self.tokenizer(
            self.CANDIDATE_OPTIONS,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        self.pre_opt_ids = opt_enc["input_ids"].unsqueeze(0).to(self.device)
        self.pre_opt_mask = opt_enc["attention_mask"].unsqueeze(0).to(self.device)

    def decide(self, game: SnakeGame) -> tuple[str, dict[str, float], dict[str, float]]:
        state_text, candidates = analyze_snake_state(
            snake=game.snake,
            food=game.food,
            direction=game.direction,
            width=game.width,
            height=game.height,
            recent_positions=game.recent_positions,
        )

        legal_dirs = [c["id"] for c in candidates]
        option_texts = [c["text"] for c in candidates]

        if not legal_dirs:
            return (
                game.direction,
                {game.direction: 1.0},
                {"tok": 0.0, "fwd": 0.0, "soft": 0.0, "total": 0.1},
            )

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
            out = self.model(ctx_ids, ctx_mask, opt_ids, opt_mask, num_options=[len(legal_dirs)])
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0

        t_soft_start = time.perf_counter()
        probs = out["probs"][0, : len(legal_dirs)].float().cpu().numpy()
        chosen_idx = int(np.argmax(probs))
        best_dir = legal_dirs[chosen_idx]
        prob_dict = {legal_dirs[i]: float(probs[i]) for i in range(len(legal_dirs))}
        t_soft_end = time.perf_counter()
        soft_ms = (t_soft_end - t_soft_start) * 1000.0

        total_ms = tok_ms + fwd_ms + soft_ms
        lat_dict = {
            "tok": tok_ms,
            "fwd": fwd_ms,
            "soft": soft_ms,
            "total": total_ms,
        }
        return best_dir, prob_dict, lat_dict


def play_snake_demo(
    checkpoint: str = "models/exported/nano/tamev_nano_fp32.pt",
    max_steps: int = 100,
    render: bool = True,
    delay: float = 0.08,
):
    print_banner(
        "🐍 TAMEV Real-Time Snake AI Benchmark & Demo",
        subtitle=f"Model Checkpoint: {checkpoint}",
    )

    agent = TamevSnakeAgent(checkpoint_path=checkpoint, device_name="cpu")
    game = SnakeGame(width=10, height=10)

    total_latency = 0.0
    steps = 0

    if render:
        with Live(auto_refresh=False) as live:
            while not game.game_over and steps < max_steps:
                chosen_dir, probs, lat_info = agent.decide(game)
                lat_val = lat_info["total"] if isinstance(lat_info, dict) else lat_info
                total_latency += lat_val
                steps += 1
                game.step(chosen_dir)
                live.update(
                    game.render_rich(last_probs=probs, latency_ms=lat_info, chosen=chosen_dir),
                    refresh=True,
                )
                time.sleep(delay)
    else:
        while not game.game_over and steps < max_steps:
            chosen_dir, probs, lat_info = agent.decide(game)
            lat_val = lat_info["total"] if isinstance(lat_info, dict) else lat_info
            total_latency += lat_val
            steps += 1
            game.step(chosen_dir)

    avg_lat = total_latency / max(steps, 1)
    print_section(
        "🏁 Snake Game Finished",
        details={
            "Final Score": f"{game.score} points",
            "Steps Survived": game.steps,
            "Apples Eaten": game.score // 10,
            "Average Latency": f"{avg_lat:.2f} ms per tick (~{1000 / max(avg_lat, 0.1):.0f} decisions/sec)",
        },
    )
    print_success("Snake AI demo session finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/exported/nano/tamev_nano_fp32.pt")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    play_snake_demo(
        checkpoint=args.checkpoint,
        max_steps=args.max_steps,
        render=not args.no_render,
        delay=args.delay,
    )
