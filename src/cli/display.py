"""CLI display utilities using Rich.

Provides beautiful console output for:
- Session progress
- Soul debate visualization
- Hypothesis cards
- Iteration curves
"""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

from src.contracts.schemas import Hypothesis, IterationTrace, SessionResult, SoulRole

console = Console()


# Soul colors for visual distinction
SOUL_COLORS = {
    SoulRole.CREATIVE: "bright_magenta",
    SoulRole.SKEPTIC: "red",
    SoulRole.METHODICAL: "blue",
    SoulRole.RISK_TAKER: "yellow",
    SoulRole.SYNTHESIZER: "green",
}

SOUL_ICONS = {
    SoulRole.CREATIVE: "🎨",
    SoulRole.SKEPTIC: "🔍",
    SoulRole.METHODICAL: "📐",
    SoulRole.RISK_TAKER: "🚀",
    SoulRole.SYNTHESIZER: "🔮",
}


def print_header(topic: str) -> None:
    """Print session header."""
    console.print()
    console.print(Panel(
        f"[bold white]Research Topic:[/bold white] {topic}",
        title="🧬 [bold cyan]Scientific Hypothesis Synthesizer[/bold cyan] 🧬",
        border_style="cyan",
    ))
    console.print()


def print_phase(phase: str, description: str = "") -> None:
    """Print a phase header."""
    console.print(f"\n[bold yellow]▶ {phase}[/bold yellow]", end="")
    if description:
        console.print(f" [dim]{description}[/dim]")
    else:
        console.print()


def print_papers_ingested(count: int, categories: list[str]) -> None:
    """Print paper ingestion summary."""
    console.print(f"  📚 Fetched [bold]{count}[/bold] papers from arXiv")
    if categories:
        console.print(f"  📂 Categories: [dim]{', '.join(categories)}[/dim]")


def print_concept_map_stats(nodes: int, edges: int, gaps: int) -> None:
    """Print concept map statistics."""
    console.print(f"  🗺️ Built concept map: [bold]{nodes}[/bold] entities, [bold]{edges}[/bold] relations, [bold]{gaps}[/bold] gaps")


def print_iteration_start(iteration: int, mode: str) -> None:
    """Print iteration start."""
    console.print(f"\n[bold blue]━━━ Iteration {iteration} ━━━[/bold blue] [dim]({mode})[/dim]")


def print_soul_message(soul: SoulRole, message: str) -> None:
    """Print a message from a soul."""
    icon = SOUL_ICONS.get(soul, "🤖")
    color = SOUL_COLORS.get(soul, "white")
    console.print(f"  {icon} [bold {color}]{soul.value.upper()}[/bold {color}]: {message}")


def print_hypothesis_card(hypothesis: Hypothesis, index: int) -> None:
    """Print a single hypothesis as a card."""
    soul_icon = SOUL_ICONS.get(hypothesis.source_soul, "📝") if hypothesis.source_soul else "📝"
    soul_color = SOUL_COLORS.get(hypothesis.source_soul, "white") if hypothesis.source_soul else "white"

    # Build card content
    content = []
    content.append(f"[bold]{hypothesis.hypothesis}[/bold]")
    content.append("")
    content.append(f"[dim]Rationale:[/dim] {hypothesis.rationale[:150]}...")
    content.append(f"[dim]Cross-domain:[/dim] {hypothesis.cross_disciplinary_connection[:100]}")
    content.append("")

    # Scores
    scores = hypothesis.scores
    score_bar = (
        f"Novelty: [{'green' if scores.novelty >= 0.7 else 'yellow'}]{scores.novelty:.0%}[/] | "
        f"Feasibility: [{'green' if scores.feasibility >= 0.6 else 'yellow'}]{scores.feasibility:.0%}[/] | "
        f"Impact: [{'green' if scores.impact >= 0.5 else 'yellow'}]{scores.impact:.0%}[/]"
    )
    content.append(score_bar)

    # Evidence
    if hypothesis.evidence.arxiv_hits > 0:
        content.append(f"[dim]arXiv hits: {hypothesis.evidence.arxiv_hits}[/dim]")

    console.print(Panel(
        "\n".join(content),
        title=f"{soul_icon} Hypothesis {index + 1}",
        border_style=soul_color,
    ))


def print_iteration_summary(trace: IterationTrace) -> None:
    """Print iteration summary."""
    console.print(f"\n  [dim]Thought:[/dim] {trace.thought}")
    console.print(f"  [dim]Action:[/dim] {trace.action}")
    console.print(f"  [dim]Result:[/dim] {trace.observation}")

    # Score indicators
    novelty_emoji = "🟢" if trace.avg_novelty >= 0.7 else "🟡" if trace.avg_novelty >= 0.4 else "🔴"
    feas_emoji = "🟢" if trace.avg_feasibility >= 0.6 else "🟡" if trace.avg_feasibility >= 0.4 else "🔴"

    console.print(f"  {novelty_emoji} Novelty: {trace.avg_novelty:.0%} | {feas_emoji} Feasibility: {trace.avg_feasibility:.0%}")


def print_progress_table(traces: list[IterationTrace]) -> None:
    """Print iteration progress as a table."""
    table = Table(title="Iteration Progress")
    table.add_column("Iter", justify="center", style="cyan")
    table.add_column("Mode", style="dim")
    table.add_column("Generated", justify="center")
    table.add_column("Surviving", justify="center")
    table.add_column("Novelty", justify="center")
    table.add_column("Feasibility", justify="center")

    for trace in traces:
        novelty_style = "green" if trace.avg_novelty >= 0.7 else "yellow" if trace.avg_novelty >= 0.4 else "red"
        feas_style = "green" if trace.avg_feasibility >= 0.6 else "yellow" if trace.avg_feasibility >= 0.4 else "red"

        table.add_row(
            str(trace.iteration),
            trace.mode_used.value[:10],
            str(trace.hypotheses_generated),
            str(trace.hypotheses_surviving),
            f"[{novelty_style}]{trace.avg_novelty:.0%}[/]",
            f"[{feas_style}]{trace.avg_feasibility:.0%}[/]",
        )

    console.print(table)


def print_final_results(result: SessionResult) -> None:
    """Print final session results."""
    console.print()
    console.print(Panel(
        f"[bold green]Session Complete[/bold green]\n\n"
        f"Stop reason: {result.stop_reason}\n"
        f"Iterations: {result.iterations_completed}\n"
        f"Papers ingested: {result.papers_ingested}\n"
        f"Final hypotheses: {len(result.final_hypotheses)}",
        title="🎉 Results",
        border_style="green",
    ))

    console.print("\n[bold]Top Hypotheses:[/bold]\n")
    for i, h in enumerate(result.final_hypotheses[:5]):
        print_hypothesis_card(h, i)


def create_progress_spinner(description: str) -> Progress:
    """Create a progress spinner."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )
