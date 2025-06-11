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

from scene_match.scene_lib.frame_matcher import FrameMatcher

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


@cli.command('build')
@click.argument('reference_video', type=click.Path(exists=True, path_type=Path))
@click.option('--sample-interval', '-s', default=10, help='Number of frames to skip when sampling')
@click.option('--start-frame', '-f', type=int, default=0, help='Start frame for analysis (default: 0)')
@click.option('--visualize/--no-visualize', '-v', default=True, help='Show visualization during processing')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output directory for index file')
def build_index(reference_video: Path, sample_interval: int = 10, start_frame: int = 0, visualize: bool = True,
                output: Path = None):
    """
    Build an index for the reference video.

    :param reference_video: Path to the reference video file
    :param sample_interval: Number of frames to skip when sampling
    :param start_frame: Start frame for analysis (default: 0)
    :param visualize: Show visualization during processing
    :param output: Output file to save the index (JSON format)
    :return: Matcher object with built index
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

            # Save results if an output file specified
            if output:
                progress.add_task("Saving index...", total=None)
                console.print(f"\nSaving results to: [blue]'{output}'[/]")
                index_location = matcher.serialize(output)
                console.print(f"\nResults saved to: [blue]'{index_location}'[/]")

    except Exception as e:
        logger.exception('')
        console.print(f"\n[red]Error: {str(e)}[/]")



@cli.command()
@click.argument('reference', type=click.Path(exists=True, path_type=Path),)  #help='reference_video / path to serialized index'
@click.argument('comparison_video', type=click.Path(exists=True, path_type=Path))
@click.option('--sample-interval', '-s', default=10, help='Number of frames to skip when sampling')
@click.option('--start-frame', '-f', type=int, default=0, help='Start frame for analysis (default: 0)')
@click.option('--visualize/--no-visualize', '-v', default=True, help='Show visualization during processing')
def analyze(reference: Path, comparison_video: Path, sample_interval: int, start_frame: int, visualize: bool, ):
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

            if reference.is_dir():
                progress.add_task("Loading matcher from index", total=None)
                matcher = FrameMatcher.deserialize(reference)
            elif reference.is_file():
                matcher = stream.get_indexed_matcher(reference,
                                                     stream_params=stream_params,
                                                     draw_params=draw_params,
                                                     index_params=index_params)
            else:
                raise ValueError(f"Invalid reference video path (not file / folder): {reference})")

            # Find matches1
            progress.add_task("Finding matches...", total=None)
            matches = stream.get_matches(comparison_video, matcher,
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
                f"Reference: [cyan]{Path(matcher.video_source).name}[/]\n"
                f"Comparison: [green]{comparison_video.name}[/]",
                title="Analysis Complete"
            ))
            console.print(table)


    except Exception as e:
        logger.exception('')
        console.print(f"\n[red]Error: {str(e)}[/]")
        raise click.Abort()


# @cli.command(name='record')
# @click.argument('source', type=click.Path(exists=True, path_type=Path))
# @click.option('--output', '-o', type=click.Path(path_type=Path), help='Output video file path')
# @click.option('--duration', '-d', type=float, default=10.0, help='Duration to record in seconds')


if __name__ == '__main__':
    cli()
