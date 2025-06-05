"""
Main entry point for SceneMatch video analysis.
"""

import rich_click as click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
import logging
import coloredlogs
import sys

# For support both python -m & python src/scenematch/__main__.py
try:
    from lib import frame_matcher as fm
except ImportError:
    parent = Path(__file__).resolve().parent.absolute()
    sys.path.insert(0, str(parent))
    from lib import frame_matcher as fm

logger = logging.getLogger(__name__)

logger_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# logger_format = '%(asctime)s [%(name)s] %(module)s::%(funcName)s %(levelname)s - %(message)s'
coloredlogs.install(level='CRITICAL', fmt=logger_format)
coloredlogs.install(level='DEBUG', fmt=logger_format, logger=logger, stream=sys.stdout, isatty=True)


# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "yellow italic"
click.rich_click.ERRORS_SUGGESTION = "Try '--help' for more information."

console = Console()


@click.group()
def cli():
    """SceneMatch - A tool for matching and analyzing drone video frames."""
    pass


@cli.command()
@click.argument('reference_video', type=click.Path(exists=True, path_type=Path))
@click.argument('comparison_video', type=click.Path(exists=True, path_type=Path))
@click.option('--sample-interval', '-s', default=10, help='Number of frames to skip when sampling')
@click.option('--visualize/--no-visualize', '-v', default=True, help='Show visualization during processing')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file for matches (JSON format)')
def analyze(reference_video: Path, comparison_video: Path, sample_interval: int, visualize: bool, output: Path):
    """
    Analyze two videos and find matching frames.
    
    REFERENCE_VIDEO: Path to the reference video file
    COMPARISON_VIDEO: Path to the video file to compare against
    """
    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
        ) as progress:
            # Initialize frame matcher
            matcher = fm.FrameMatcher(reference_video,  sample_interval=sample_interval, visualize=visualize)

            # Build index for reference video
            progress.add_task("Building index for reference video...", total=None)
            matcher.build_index()

            # Find matches
            progress.add_task("Finding matches...", total=None)
            matches = matcher.find_matches(comparison_video, "comparison")

            # Display results
            table = Table(title="Matching Results")
            table.add_column("Reference Frame", justify="right", style="cyan")
            table.add_column("Reference Time", justify="right", style="cyan")
            table.add_column("Comparison Frame", justify="right", style="green")
            table.add_column("Comparison Time", justify="right", style="green")
            table.add_column("Distance Score", justify="right", style="yellow")

            for match in matches:
                table.add_row(
                    str(match.frame_reference.frame_index),
                    f"{match.frame_reference.timestamp:.2f}s",
                    str(match.frame.frame_index),
                    f"{match.frame.timestamp:.2f}s",
                    f"{match.distance_score:.4f}"
                )

            console.print(Panel.fit(
                f"Found [bold green]{len(matches)}[/] matches between videos\n"
                f"Reference: [cyan]{reference_video.name}[/]\n"
                f"Comparison: [green]{comparison_video.name}[/]",
                title="Analysis Complete"
            ))
            console.print(table)

            # Save results if an output file specified
            if output:
                import json
                from datetime import datetime

                results = {
                    "timestamp": datetime.now().isoformat(),
                    "reference_video": str(reference_video),
                    "comparison_video": str(comparison_video),
                    "sample_interval": sample_interval,
                    "matches": [
                        {
                            "reference_frame": match.frame_reference.frame_index,
                            "reference_time": match.frame_reference.timestamp,
                            "comparison_frame": match.frame.frame_index,
                            "comparison_time": match.frame.timestamp,
                            "similarity_score": match.distance_score
                        }
                        for match in matches
                    ]
                }

                output.write_text(json.dumps(results, indent=2))
                console.print(f"\nResults saved to: [blue]{output}[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis stopped by user[/]")
    except Exception as e:
        logger.exception('')
        console.print(f"\n[red]Error: {str(e)}[/]")
        raise click.Abort()


if __name__ == '__main__':
    cli()
