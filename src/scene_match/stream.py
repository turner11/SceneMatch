import itertools
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import reactivex as rx
from reactivex import operators as ops
from rich.console import Console
from .scene_lib import frame_matcher as fm
from .imaging import reader
from .imaging.stream_types import StreamParams, IndexParams, FrameCapture, DrawParams
from .scene_lib.visualizer import Visualizer

logger = logging.getLogger(__name__)
ESCAPE_KEY = 27  # ESC key
QUIT_KEY = ord('q')  # 'q' key to quit
PAUSE_CHAR = ' '  # Space key to toggle pause/resume
TOGGLE_MATCH_DRAW_CHAR = ']'  # 'm' key to toggle match drawing
TOGGLE_DESCRIPTION_CHAR = '['  # 'd' key to toggle description drawing
MORE_FEATURES_CHAR, LESS_FEATURES_CHAR = '+', '-'
SLOWER_CHAR, FASTER_CHAR = ',', '.'


def get_frames(source, stream_params: StreamParams):
    stream = reader.get_stream(source, stream_params=stream_params)
    observable = rx.from_iterable(stream).pipe(
        # Share a single subscription among multiple observers
        # (This is an alias for a composed publish() and ref_count().)
        ops.share(),
    )
    return observable


def get_features_pipeline(source,
                          matcher: fm.FrameMatcher,
                          stream_params: StreamParams = None,
                          draw_params: DrawParams = None,
                          pre_pipes=tuple()):
    stream_params = stream_params or StreamParams()
    draw_params = draw_params or DrawParams()
    visualize = draw_params.visualize

    if isinstance(source, rx.observable.observable.Observable):
        # Use a pre-built observable
        observable_base = source
    else:
        # build observable from source
        observable_base = get_frames(source, stream_params=stream_params)

    if pre_pipes:
        # Apply pre-pipes
        observable_base = observable_base.pipe(*pre_pipes)

    console = Console()

    def handle_key_pressed(_d):
        image, frame_data, pressed_key = [_d[k] for k in ('image', 'frame_data', 'key_pressed')]
        pressed_char = chr(pressed_key)
        if not pressed_char or pressed_key == 255:
            return
        sp, dp = stream_params, draw_params
        speed_step_size = 5
        features_step_size = 25
        # noinspection PyUnreachableCode
        match (pressed_key, pressed_char):
            case _, k if k == FASTER_CHAR:
                sp.sample_interval += speed_step_size
                sp.sample_interval = min(sp.sample_interval, 100)  # Limit max sample interval
                console.print(f'Increased frame sample to {stream_params.sample_interval}')
            case _, k if k == SLOWER_CHAR:
                sp.sample_interval -= speed_step_size
                sp.sample_interval = max(sp.sample_interval, 1)  # Limit min sample interval
                console.print(f'Decreased frame sample to {stream_params.sample_interval}')
            case _, k if k == TOGGLE_DESCRIPTION_CHAR:
                dp.show_keypoints = not dp.show_keypoints
                console.print(f'Toggled description drawing to {dp.show_keypoints}')
            case _, k if k == TOGGLE_MATCH_DRAW_CHAR:
                dp.show_matches = not dp.show_matches
                console.print(f'Toggled match drawing to {dp.show_matches}')
            case _, k if k == PAUSE_CHAR:  # pause / resume
                while chr(cv2.waitKey(1) & 0xFF) != PAUSE_CHAR:
                    pass
                _d.update(pressed_key='')
            case _, k if k == MORE_FEATURES_CHAR:
                draw_params.n_features += features_step_size
                console.print(f'Increased number of features to {draw_params.n_features}')
            case _, k if k == LESS_FEATURES_CHAR:
                draw_params.n_features -= features_step_size
                draw_params.n_features = max(draw_params.n_features, -1)
                console.print(f'Decreased number of features to {draw_params.n_features}')
            case _:
                # console.print(f'[red]Pressed: key: {pressed_key}, char: {pressed_char}[/red]')
                pass

    counter = itertools.count()

    # noinspection PyArgumentList
    observable = observable_base.pipe(
        ops.map(
            lambda f: defaultdict(lambda: None, **{'frame': f,
                                                   'image': f.image.copy(),
                                                   'i': next(counter)}), ),
        ops.do_action(lambda d: d.update({'key_pressed': cv2.waitKey(1) & 0xFF})),

        ops.take_while(lambda d: d.get('key_pressed') != ESCAPE_KEY),
        ops.take_while(lambda d: d.get('key_pressed') != QUIT_KEY),

        ops.do_action(lambda d: handle_key_pressed(d)),
        ops.filter(lambda d: d['image'].size > 0),
        # Handle new frame
        # ops.do_action(lambda d: d.update(detections=detector.detect(d['image'])) if d['detect'] else None),
        ops.do_action(lambda d: d.update({'frame_data': matcher.save_frame_features(d['image'], d['frame'].n)})),
        ops.share(),
    )

    if visualize:
        visualizer = Visualizer(water_mark='Frame Features', )

        def draw_frame(_d):
            dp = draw_params
            image, frame_data = [_d[k] for k in ('image', 'frame_data',)]
            show_keypoints = dp.show_keypoints
            n_keypoints = draw_params.n_features

            keypoints = frame_data.keypoints
            visualizer.show_frame(image, keypoints, show_keypoints, n_keypoints)

        observable = observable.pipe(
            ops.do_action(lambda d: draw_frame(_d=d)),
        )

    return observable


def get_indexed_matcher(source,
                        stream_params: StreamParams = None,
                        draw_params: DrawParams = None,
                        index_params: IndexParams = None,
                        pre_pipes=tuple()):
    index_params = index_params or IndexParams()
    matcher = fm.FrameMatcher(source, n_features=index_params.n_features)

    obs = get_features_pipeline(source, matcher, stream_params=stream_params, draw_params=draw_params,
                                pre_pipes=pre_pipes)
    frame_metadata_by_frame_index = {}

    def handle_error(_e):
        logger.exception('Got an error while collecting features')
        raise _e

    def handle_completion():
        logger.info('Feature Collection completed')

    obs = obs.pipe(ops.map(lambda d: d['frame_data']), )

    try:
        obs.subscribe(
            on_next=lambda frame_data: frame_metadata_by_frame_index.update({frame_data.frame_index: frame_data}),
            on_error=(lambda e: handle_error(e)),
            on_completed=lambda: handle_completion()

        )
    except KeyboardInterrupt:
        logger.info("Feature Collection stopped by user")

    matcher.build_index()
    return matcher



def get_matches(source, matcher, stream_params: StreamParams = None, draw_params: DrawParams = None):
    # copy stream_params
    reference_source = matcher.video_source
    orig_visualize = draw_params.visualize
    draw_params.visualize = False
    obs = get_features_pipeline(source, matcher, stream_params=stream_params, draw_params=draw_params)

    obs = obs.pipe(
        ops.do_action(lambda d: d.update({'match': matcher.match(d['frame_data'])})),
    )

    draw_params.visualize = orig_visualize

    console = Console()

    matches = []
    visualizer, cap = None, None
    if draw_params.visualize:
        visualizer = Visualizer(water_mark='Frame Matches', )
        cap = cv2.VideoCapture(str(reference_source))

    def _handle_next(data):
        image, match = [data[k] for k in ('image', 'match')]
        match: fm.FrameMatch
        image: np.ndarray

        matches.append(match)
        if visualizer is not None:
            frame_meta, reference_meta = match.frame, match.frame_reference
            my_index, reference_index = frame_meta.frame_index, reference_meta.frame_index
            cap.set(cv2.CAP_PROP_POS_FRAMES, reference_index)
            ret_ref, image_ref = cap.read()
            if ret_ref:
                # Create matches for visualization
                visualizer.show_frame_matches(image, image_ref, match,
                                              draw_params.show_keypoints,
                                              draw_params.show_matches,
                                              draw_params.n_features)

    def _on_complete():
        if visualizer is not None:
            visualizer.close()
        if cap is not None:
            cap.release()
        console.print(f'[green]Collected {len(matches)} matches[/green]')

    def _on_error(e):
        console.print(f'[red]Error: {e}')
        raise e

    obs.subscribe(on_next=_handle_next,
                  on_completed=_on_complete,
                  on_error=_on_error)

    return matches


def get_write_video_action(output: str | Path):
    output = Path(output)
    aggregator = {}

    def _write_video(frame: FrameCapture):
        image = frame.image
        writer = aggregator.get('writer')
        if writer is None:
            # noinspection PyUnresolvedReferences
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            shape = image.shape[1], image.shape[0]
            writer = cv2.VideoWriter(str(output), fourcc, frame.stream_fps, shape)
            aggregator['writer'] = writer

        writer.write(image)
        cv2.waitKey(1)

    return _write_video


def record(source, output: str | Path, duration: int | float, draw_params: DrawParams = None):
    aggregator = {'count': 0, 'source_fps': float('inf')}
    draw_params = draw_params or DrawParams(visualize=True)
    console = Console()

    def _handle_next(frame: FrameCapture):
        image = frame.image
        if draw_params.visualize:
            cv2.imshow(f'frame (Source: {source})', image)

    def _on_complete():
        writer = aggregator.get('writer')
        if writer:
            # noinspection PyUnresolvedReferences
            writer.release()

    def _on_error(e):
        console.print(f'[red]Error: {e}')
        raise e

    write_video_action = get_write_video_action(output)

    # video_stream = get_frames(source)
    stream_params = StreamParams()
    matcher = fm.FrameMatcher()
    video_stream = get_features_pipeline(source, matcher, stream_params=stream_params, draw_params=draw_params)
    pipeline = video_stream.pipe(
        ops.do_action(lambda _: aggregator.update(count=aggregator['count'] + 1)),
        ops.do_action(lambda frame: aggregator.update(source_fps=frame.stream_fps)),
        ops.do_action(lambda _: aggregator.update(elapsed=aggregator['count'] / aggregator['source_fps'])),
        ops.take_while(lambda _: aggregator['elapsed'] < duration),
        ops.do_action(lambda frame: write_video_action(frame))
    )

    pipeline.subscribe(on_next=_handle_next,
                       on_completed=_on_complete,
                       on_error=_on_error)
