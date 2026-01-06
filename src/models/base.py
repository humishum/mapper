"""Base model interface for reconstruction models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple
import logging

from ..core.types import ReconstructionResult, VideoInput

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all reconstruction models.

    Each model must implement:
    - load() - Load model weights/initialize
    - reconstruct() - Run reconstruction on video input

    Optional override:
    - get_default_config() - Return default configuration
    """

    # Model identifier
    name: str = "base"

    # Capability flags (informational - used for experiment tracking)
    outputs_metric_scale: bool = False  # Does the model output metric-scale coordinates?
    outputs_poses: bool = False  # Does the model output camera poses?
    outputs_confidence: bool = False  # Does the model output per-point confidence?
    supports_video_input: bool = False  # Can the model process video directly?

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize model with configuration.

        Args:
            config: Model-specific configuration dictionary.
                   If None, uses get_default_config().
        """
        self.config = config or self.get_default_config()
        self.model = None
        self._is_loaded = False

    @abstractmethod
    def load(self, weights_path: Optional[Path] = None) -> None:
        """
        Load model weights and prepare for inference.

        Args:
            weights_path: Optional path to model weights.
                         If None, uses path from config or downloads.
        """
        pass

    @abstractmethod
    def reconstruct(
        self,
        video_input: VideoInput,
        output_dir: Path,
    ) -> ReconstructionResult:
        """
        Run 3D reconstruction on video input.

        Args:
            video_input: Processed video with extracted frames and telemetry
            output_dir: Directory to save model outputs (PLY files, etc.)

        Returns:
            ReconstructionResult containing point cloud and optional poses
        """
        pass

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Return default configuration for this model.

        Override in subclasses to provide model-specific defaults.
        """
        return {}

    def ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if not self._is_loaded:
            logger.info(f"Loading {self.name} model...")
            self.load()
            self._is_loaded = True

    def build_windows(
        self,
        total_images: int,
        use_chunking: bool,
        window_size: Optional[int],
        window_overlap: Optional[int],
    ) -> List[Tuple[int, int]]:
        """Build sliding windows for chunked reconstruction."""
        if not use_chunking or window_size is None:
            return [(0, total_images)]

        window_size = int(window_size)
        window_overlap = int(window_overlap or 0)

        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_overlap >= window_size:
            raise ValueError("window_overlap must be smaller than window_size")

        step = window_size - window_overlap
        windows = []
        start = 0
        while start < total_images:
            end = min(start + window_size, total_images)
            windows.append((start, end))
            if end == total_images:
                break
            start += step
        return windows

    def get_capabilities(self) -> dict:
        """Return model capabilities as a dictionary."""
        return {
            "name": self.name,
            "outputs_metric_scale": self.outputs_metric_scale,
            "outputs_poses": self.outputs_poses,
            "outputs_confidence": self.outputs_confidence,
            "supports_video_input": self.supports_video_input,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, loaded={self._is_loaded})"
