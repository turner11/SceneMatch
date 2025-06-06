import itertools
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import reactivex as rx
from reactivex import operators as ops
from rich.console import Console
from .lib import frame_matcher as fm
from .imaging import reader
from .imaging.stream_types import StreamParams, IndexParams, FrameCapture
from .lib.visualizer import Visualizer

logger = logging.getLogger(__name__)
ESCAPE_KEY = 27  # ESC key


def get_frames(source, start_frame=0, sample_interval=1):
    stream = reader.get_stream(source, start_frame=start_frame, sample_interval=sample_interval)
    observable = rx.from_iterable(stream).pipe(
        # Share a single subscription among multiple observers
        # (This is an alias for a composed publish() and ref_count().)
        ops.share(),
    )
    return observable


def get_features_pipeline(source,
                          matcher: fm.FrameMatcher,
                          stream_params: StreamParams = None,
                          pre_pipes=tuple(), quit_key=ESCAPE_KEY):
    stream_params = stream_params or StreamParams()
    sample_interval = stream_params.sample_interval
    visualize = stream_params.visualize

    if isinstance(source, rx.observable.observable.Observable):
        # Use a pre-built observable
        observable_base = source
    else:
        # build observable from source
        observable_base = get_frames(source, start_frame=stream_params.start_frame, sample_interval=sample_interval)

    if pre_pipes:
        # Apply pre-pipes
        observable_base = observable_base.pipe(*pre_pipes)

    counter = itertools.count()

    # noinspection PyArgumentList
    observable = observable_base.pipe(
        ops.map(
            lambda f: defaultdict(lambda: None, **{'frame': f,
                                                   'image': f.image.copy(),
                                                   'i': next(counter)}), ),
        ops.do_action(lambda d: d.update({'key_pressed': cv2.waitKey(1) & 0xFF})),
        ops.take_while(lambda d: d.get('pressed_key') != quit_key),
        ops.filter(lambda d: d['image'].size > 0),
        # Handle new frame
        # ops.do_action(lambda d: d.update(detections=detector.detect(d['image'])) if d['detect'] else None),
        ops.do_action(lambda d: d.update({'frame_data': matcher.save_frame_features(d['image'], d['frame'].n)})),
        ops.share(),
    )

    if visualize:
        visualizer = Visualizer()
        observable = observable.pipe(
            ops.do_action(lambda d: visualizer.show_frame(d['image'], d['frame_data'].keypoints)),
        )

    return observable


def get_indexed_matcher(source,
                        stream_params: StreamParams = None,
                        index_params: IndexParams = None,
                        pre_pipes=tuple()):
    index_params = index_params or IndexParams()
    matcher = fm.FrameMatcher(n_features=index_params.n_features)

    obs = get_features_pipeline(source, matcher, stream_params=stream_params, pre_pipes=pre_pipes)
    frame_metadata_by_frame_index = {}

    def handle_error(_e):
        logger.exception('Got an error while collecting features')
        raise _e

    obs = obs.pipe(ops.map(lambda d: d['frame_data']), )

    try:
        obs.subscribe(
            on_next=lambda frame_data: frame_metadata_by_frame_index.update({frame_data.frame_index: frame_data}),
            on_error=(lambda e: handle_error(e)),
            on_completed=lambda: matcher.build_index()

        )
    except KeyboardInterrupt:
        logger.info("Feature Collection stopped by user")

    matcher.build_index()
    return matcher


def get_matches(source, reference_source, matcher, stream_params: StreamParams = None, ):
    # copy stream_params
    pipeline_stream_params = StreamParams(**stream_params.__dict__) if stream_params else StreamParams()
    pipeline_stream_params.visualize = False
    obs = get_features_pipeline(source, matcher, stream_params=pipeline_stream_params)

    obs = obs.pipe(
        ops.do_action(lambda d: d.update({'match': matcher.match(d['frame_data'])})),
    )

    console = Console()

    matches = []
    visualizer, cap = None, None
    if stream_params.visualize:
        visualizer = Visualizer()
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
                visualizer.show_matches(image, image_ref, match)

    def _on_complete():
        ...

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


def record(source, output: str | Path, duration: int | float, show: bool):
    aggregator = {'count': 0, 'source_fps': float('inf')}
    console = Console()

    def _handle_next(frame: FrameCapture):
        image = frame.image
        if show:
            cv2.imshow(f'frame (Source: {source})', image)

    def _on_complete():
        writer = aggregator.get('writer')
        if writer:
            # noinspection PyUnresolvedReferences
            writer.release()
        cv2.destroyAllWindows()

    def _on_error(e):
        console.print(f'[red]Error: {e}')
        raise e

    write_video_action = get_write_video_action(output)

    # video_stream = get_frames(source)
    stream_params = StreamParams(visualize=True)
    matcher = fm.FrameMatcher()
    video_stream = get_features_pipeline(source, matcher, stream_params)
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
