from dataclasses import dataclass
from pathlib import Path
import json
from typing import Sequence

import cv2
import numpy as np


@dataclass
class FrameMetadata:
    """Metadata for a video frame."""
    frame_index: int
    timestamp: float
    keypoints: tuple[cv2.KeyPoint, ...]
    features: np.array
    video_id: str = ''

    def to_dict(self) -> dict:
        """Convert frame metadata to a dictionary."""
        # Convert keypoints to a serializable format
        keypoints_data = [{
            'pt': [float(kp.pt[0]), float(kp.pt[1])],
            'size': float(kp.size),
            'angle': float(kp.angle),
            'response': float(kp.response),
            'octave': int(kp.octave),
            'class_id': int(kp.class_id)
        } for kp in self.keypoints]

        # Convert numpy array to list
        features_data = self.features.tolist()

        return {
            'frame_index': self.frame_index,
            'timestamp': self.timestamp,
            'keypoints': keypoints_data,
            'features': features_data,
            'video_id': self.video_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FrameMetadata':
        """Create frame metadata from a dictionary."""
        # Convert keypoints data back to cv2.KeyPoint objects
        keypoints = tuple(
            cv2.KeyPoint(
                x=float(kp['pt'][0]),
                y=float(kp['pt'][1]),
                size=float(kp['size']),
                angle=float(kp['angle']),
                response=float(kp['response']),
                octave=int(kp['octave']),
                class_id=int(kp['class_id'])
            )
            for kp in data['keypoints']
        )

        # Convert the feature list back to a numpy array
        features = np.array(data['features']).astype(np.float32)

        return cls(
            frame_index=data['frame_index'],
            timestamp=data['timestamp'],
            keypoints=keypoints,
            features=features,
            video_id=data['video_id']
        )

    def serialize(self, path: str | Path):
        """
        Serialize the frame metadata to a JSON file.
        
        Args:
            path: Path to save the serialized data
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def deserialize(path: str | Path) -> 'FrameMetadata':
        """
        Deserialize frame metadata from a JSON file.
        
        Args:
            path: Path to the serialized data file
            
        Returns:
            FrameMetadata: The deserialized frame metadata
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return FrameMetadata.from_dict(data)


@dataclass
class FrameMatch:
    """Represents a match between two frames."""
    frame: FrameMetadata
    frame_reference: FrameMetadata
    distance_score: float
    features_percentage: float
    notes: str = ''

    def to_dict(self) -> dict:
        """Convert frame match to a dictionary."""
        return {
            'frame': self.frame.to_dict(),
            'frame_reference': self.frame_reference.to_dict(),
            'distance_score': float(self.distance_score),
            'features_percentage': float(self.features_percentage),
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FrameMatch':
        """Create frame match from a dictionary."""
        return cls(
            frame=FrameMetadata.from_dict(data['frame']),
            frame_reference=FrameMetadata.from_dict(data['frame_reference']),
            distance_score=float(data['distance_score']),
            features_percentage=float(data['features_percentage']),
            notes=data['notes']
        )

    def serialize(self, path: str | Path):
        """
        Serialize the frame match to a JSON file.
        
        Args:
            path: Path to save the serialized data
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def deserialize(path: str | Path) -> 'FrameMatch':
        """
        Deserialize frame match from a JSON file.
        
        Args:
            path: Path to the serialized data file
            
        Returns:
            FrameMatch: The deserialized frame match
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return FrameMatch.from_dict(data)

    @staticmethod
    def serialize_matches(path: str | Path, matches: Sequence['FrameMatch']):
        """
        Serialize multiple frame matches to a JSON file.
        
        Args:
            path: Path to save the serialized data
            matches: Sequence of FrameMatch objects to serialize
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        dicts = [m.to_dict() for m in matches]
        with open(path, 'w') as f:
            json.dump(dicts, f, indent=2)

    @staticmethod
    def deserialize_matches(path: str | Path) -> tuple['FrameMatch', ...]:
        """
        Deserialize multiple frame matches from a JSON file.
        
        Args:
            path: Path to the serialized data file
            
        Returns:
            tuple[FrameMatch, ...]: The deserialized frame matches
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return tuple(FrameMatch.from_dict(m) for m in data)


class FeaturesParams:
    n_features: int = 1000
    max_distance: int = 500
