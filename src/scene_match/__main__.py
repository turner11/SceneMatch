"""
Main entry point for SceneMatch video analysis.
"""

# import cv2
# while True:
#     k = cv2.waitKey(500)
#     kf = k & 0xFF
#     c = chr(kf)
#     if k == -1:
#         ...
#     print(f'k: {k}')
#     print(f'kf: {kf}')
#     print(f'c: {c}')

import rich_click as click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
import logging
import coloredlogs
import sys

from scene_match.imaging.stream_types import DrawParams

# For support both python -m & python src/scenematch/__main__.py
try:
    from scene_match import stream
except ImportError:
    parent = Path(__file__).resolve().parent.absolute()
    sys.path.insert(0, str(parent))
    from scene_match import stream

from scene_match.imaging import stream_types

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
@click.option('--start-frame', '-f', type=int, default=0, help='Start frame for analysis (default: 0)')
@click.option('--visualize/--no-visualize', '-v', default=True, help='Show visualization during processing')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file for matches (JSON format)')
def analyze(reference_video: Path, comparison_video: Path, sample_interval: int, start_frame, visualize: bool,
            output: Path):
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
            stream_params = stream_types.StreamParams(start_frame=start_frame, sample_interval=sample_interval)
            draw_params = stream_types.DrawParams(visualize=visualize)
            index_params = stream_types.IndexParams()

            matcher = stream.get_indexed_matcher(reference_video,
                                                 stream_params=stream_params,
                                                 draw_params=draw_params,
                                                 index_params=index_params)

            # Find matches1
            progress.add_task("Finding matches...", total=None)

            matches = stream.get_matches(comparison_video, reference_video, matcher,
                                         stream_params=stream_params,
                                         draw_params=draw_params, )

            # Display results
            table = Table(title="Matching Results")
            table.add_column("Comparison Frame", justify="right", style="green")
            table.add_column("Comparison Time", justify="right", style="green")
            table.add_column("Reference Frame", justify="right", style="cyan")
            table.add_column("Reference Time", justify="right", style="cyan")
            table.add_column("Distance Score", justify="right", style="yellow")
            table.add_column("Features Percentage", justify="right", style="yellow")
            table.add_column("Notes", justify="left", style="magenta")

            for match in matches:
                table.add_row(
                    str(match.frame.frame_index),
                    f"{match.frame.timestamp:.2f}s",
                    str(match.frame_reference.frame_index),
                    f"{match.frame_reference.timestamp:.2f}s",
                    f"{match.distance_score:.4f}",
                    f"{match.features_percentage:.2f}%",
                    f"{match.notes}"
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
    except Exception as e:
        logger.exception('')
        console.print(f"\n[red]Error: {str(e)}[/]")
        raise click.Abort()


@cli.command(name='build')
@click.argument('reference_video', type=click.Path(exists=True, path_type=Path))
@click.option('--sample-interval', '-s', default=10, help='Number of frames to skip when sampling')
@click.option('--visualize/--no-visualize', '-v', default=True, help='Show visualization during processing')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file for index (JSON format)')
def build_index(reference_video: Path, sample_interval: int, visualize: bool, output: Path = None):
    """
    Build an index for the reference video.

    reference_video: Path to the reference video file
    sample_interval: Number of frames to skip when sampling
    visualize: Show visualization during processing
    output: output file to save the index (JSON format)
    """
    try:

        stream_params = stream_types.StreamParams(sample_interval=sample_interval)
        draw_params = DrawParams(visualize=visualize)
        index_params = stream_types.IndexParams()
        matcher = stream.get_indexed_matcher(reference_video,
                                             stream_params=stream_params,
                                             draw_params=draw_params,
                                             index_params=index_params)
        console.print(f"[green]Index built successfully for {reference_video.name}[/]")
        return matcher
    except Exception as e:
        logger.exception('')
        console.print(f"[red]Error building index: {str(e)}[/]")
        raise click.Abort()


if __name__ == '__main__':
    cli()
