"""Main CLI entry point for the Scientific Hypothesis Synthesizer."""

import asyncio
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.cli.display import (
    print_concept_map_stats,
    print_final_results,
    print_header,
    print_iteration_start,
    print_iteration_summary,
    print_papers_ingested,
    print_phase,
    print_progress_table,
)
from src.contracts.schemas import RalphConfig
from src.ralph.orchestrator import RalphOrchestrator

load_dotenv()

app = typer.Typer(
    name="novelist",
    help="Scientific Hypothesis Synthesizer - An AI research collective",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    topic: str = typer.Argument(..., help="Research topic or question"),
    max_iterations: int = typer.Option(20, "--max-iterations", "-i", help="Maximum iterations"),
    max_cost: float = typer.Option(5.0, "--max-cost", "-c", help="Maximum cost in USD"),
    max_time: int = typer.Option(600, "--max-time", "-t", help="Maximum runtime in seconds"),
    output_dir: Path = typer.Option(Path("./sessions"), "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run a hypothesis generation session.

    Example:
        novelist run "CRISPR delivery mechanisms for neural tissue"
    """
    # Check for API key
    import os
    if not os.getenv("GEMINI_API_KEY"):
        console.print("[red]Error:[/red] GEMINI_API_KEY not set.")
        console.print("Set it in your environment or .env file.")
        raise typer.Exit(1)

    # Create config
    config = RalphConfig(
        max_iterations=max_iterations,
        max_cost_usd=max_cost,
        max_runtime_seconds=max_time,
    )

    # Print header
    print_header(topic)

    # Run the orchestrator
    async def _run() -> None:
        orchestrator = RalphOrchestrator(config)

        # Hook into orchestrator for progress display
        # (In a more complete implementation, we'd use callbacks)

        print_phase("Initializing", "Setting up research session...")

        result = await orchestrator.run(topic, output_dir)

        # Print progress table
        if result.traces:
            print_progress_table(result.traces)

        # Print final results
        print_final_results(result)

        # Print save location
        session_dir = output_dir / result.session_id
        console.print(f"\n[dim]Session saved to: {session_dir}[/dim]")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def test_arxiv(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(5, "--max", "-n", help="Maximum results"),
) -> None:
    """Test arXiv API connection.

    Example:
        novelist test-arxiv "machine learning"
    """
    from src.kb.arxiv_client import ArxivClient

    async def _test() -> None:
        console.print(f"Searching arXiv for: [bold]{query}[/bold]")
        async with ArxivClient() as client:
            papers = await client.search(query, max_results=max_results)

        console.print(f"\nFound {len(papers)} papers:\n")
        for i, paper in enumerate(papers, 1):
            console.print(f"[bold cyan]{i}.[/bold cyan] [{paper.arxiv_id}] {paper.title[:70]}...")
            console.print(f"   [dim]Categories: {', '.join(paper.categories[:3])}[/dim]")

    asyncio.run(_test())


@app.command()
def test_souls(
    topic: str = typer.Argument("quantum computing in biology", help="Research topic"),
) -> None:
    """Test soul generation (Creative only).

    Example:
        novelist test-souls "CRISPR applications"
    """
    import os
    if not os.getenv("GEMINI_API_KEY"):
        console.print("[red]Error:[/red] GEMINI_API_KEY not set.")
        raise typer.Exit(1)

    from src.contracts.schemas import GenerationMode
    from src.soul.collective import SoulCollective

    async def _test() -> None:
        console.print(f"Testing soul generation for: [bold]{topic}[/bold]\n")

        collective = SoulCollective()
        context = {
            "gaps": [],
            "contradictions": [],
            "high_freq_entities": [],
            "lessons": [],
        }

        console.print("[dim]Calling Creative soul...[/dim]")
        hypotheses = await collective.quick_generate(
            topic=topic,
            context=context,
            mode=GenerationMode.GAP_HUNT,
        )

        console.print(f"\nGenerated {len(hypotheses)} hypotheses:\n")
        for i, h in enumerate(hypotheses, 1):
            console.print(f"[bold green]{i}.[/bold green] {h.hypothesis}")
            console.print(f"   [dim]Keywords: {', '.join(h.novelty_keywords[:3])}[/dim]\n")

    asyncio.run(_test())


@app.command()
def version() -> None:
    """Show version information."""
    from src import __version__
    console.print(f"[bold]Novelist[/bold] v{__version__}")
    console.print("Scientific Hypothesis Synthesizer")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
