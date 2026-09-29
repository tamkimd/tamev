# decision_engine/utils/console.py
"""
Standardized Console and Logging Utilities for TAMEV using Rich.
Provides beautiful, clear, and modern terminal styling across all TAMEV pipelines.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Global console instance
_console: Console | None = None


def get_console() -> Console:
    """Returns the shared Console instance, creating it if needed."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def print_banner(
    title: str,
    subtitle: str | None = None,
    width: int = 70,
    char: str = "=",
) -> None:
    """Prints a beautiful rounded Panel banner with Rich."""
    console = get_console()
    content = Text()
    content.append(f"{title}\n", style="bold cyan")
    if subtitle:
        content.append(f"  {subtitle}", style="dim white")

    panel = Panel(
        content,
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    )
    console.print(panel)


def print_section(
    title: str,
    details: dict[str, Any] | list[tuple[str, Any]] | None = None,
    width: int = 70,
    char: str = "-",
) -> None:
    """Prints a section divider with an optional key-value summary."""
    console = get_console()
    console.print()
    console.print(Rule(title=f"[bold cyan]{title}[/]", style="cyan", characters=char))
    if details:
        print_kv_list(details)


def print_kv_list(
    items: dict[str, Any] | list[tuple[str, Any]],
    indent: int = 2,
    key_width: int = 24,
) -> None:
    """Prints aligned and styled key-value pairs."""
    console = get_console()
    pairs = items.items() if isinstance(items, dict) else items
    pad = " " * indent
    for k, v in pairs:
        key_text = f"{pad}{k!s:<{key_width}}"
        console.print(f"[cyan]{key_text}[/] : [bold white]{v}[/]")


def format_metric(val: float | None, is_pct: bool = False, decimals: int = 2) -> str:
    """Standardizes float metric formatting."""
    if val is None:
        return "N/A"
    if is_pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"


def print_metrics_summary(title: str, metrics: dict[str, Any]) -> None:
    """Prints a standardized, beautiful Rich table for metrics."""
    console = get_console()

    table = Table(
        title=f"[bold magenta]{title}[/]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
        show_edge=True,
    )
    table.add_column("Metric", style="cyan", min_width=24)
    table.add_column("Value", style="bold white", justify="right", min_width=18)
    table.add_column("Status / Assessment", style="italic green", min_width=20)

    for k, v in metrics.items():
        status = "[dim]Logged[/]"
        if isinstance(v, float):
            if "acc" in k.lower() or "rate" in k.lower():
                val_str = f"{v * 100:.2f}%"
                status = "[bold green]Optimal[/]" if v > 0.50 else "[yellow]Good[/]"
            elif "drift" in k.lower():
                val_str = f"{v:.8f}"
                status = (
                    "[bold green]Exact Invariance[/]" if v == 0.0 else "[bold red]Drift Detected[/]"
                )
            elif "lat" in k.lower() or "ms" in k.lower():
                val_str = f"{v:.2f} ms"
                status = "[bold green]< 10ms SLA Met[/]" if v < 10.0 else "[yellow]Sub-optimal[/]"
            elif "ece" in k.lower() or "brier" in k.lower():
                val_str = f"{v:.4f}"
                status = "[bold green]Calibrated[/]" if v < 0.05 else "[yellow]Review[/]"
            else:
                val_str = f"{v:.4f}"
        else:
            val_str = str(v)

        table.add_row(k, val_str, status)

    console.print()
    console.print(table)


def print_scorecard_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    """Renders a complete benchmark or model scorecard table."""
    console = get_console()
    table = Table(
        title=f"[bold green]{title}[/]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold bright_cyan",
        show_lines=True,
    )
    for h in headers:
        table.add_column(h, justify="center")

    for row in rows:
        table.add_row(*row)

    console.print()
    console.print(table)


def create_table(
    title: str | None = None,
    headers: list[str] | None = None,
    box_style: Any = box.ROUNDED,
    border_style: str = "cyan",
    header_style: str = "bold bright_cyan",
) -> Table:
    """Creates and returns a standardized Rich Table."""
    table = Table(
        title=f"[bold cyan]{title}[/]" if title else None,
        box=box_style,
        border_style=border_style,
        header_style=header_style,
    )
    if headers:
        for h in headers:
            table.add_column(h)
    return table


def print_success(msg: str) -> None:
    """Prints a styled success message."""
    get_console().print(f"[bold green]✔ [SUCCESS][/] {msg}")


def print_warning(msg: str) -> None:
    """Prints a styled warning message."""
    get_console().print(f"[bold yellow]⚠ [WARNING][/] {msg}")


_stderr_console: Console | None = None


def get_stderr_console() -> Console:
    """Returns the shared stderr Console instance, creating it if needed."""
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(stderr=True)
    return _stderr_console


def print_error(msg: str) -> None:
    """Prints a styled error message."""
    get_stderr_console().print(f"[bold red]✖ [ERROR][/] {msg}")


def print_info(msg: str) -> None:
    """Prints a styled informational message."""
    get_console().print(f"[bold cyan]ℹ [INFO][/] {msg}")
