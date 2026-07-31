"""Model wrappers for 3D reconstruction."""

from typing import Type
import importlib
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)

# Model registry - maps model names to their module paths
# Format: "name": "module.path.ClassName"
_MODEL_REGISTRY = {
    "must3r": "src.models.must3r.MUSt3RModel",
    "vggt": "src.models.vggt.VGGTModel",
    "da3_streaming": "src.models.da3_streaming.DA3StreamingModel",
    "vggt_long": "src.models.vggt_long.VGGTLongModel",
    "mast3r_slam": "src.models.mast3r_slam.MASt3RSLAMModel",
    "vggt_omega": "src.models.vggt_omega.VGGTOmegaModel",
    "orb_slam": "src.models.orb_slam.ORBSLAMModel",
}


def get_model(name: str) -> Type[BaseModel]:
    """
    Get a model class by name.

    Args:
        name: Registered model name (see :func:`list_models`)

    Returns:
        Model class (not instantiated)

    Raises:
        ValueError: If model name is not found

    Example:
        >>> ModelClass = get_model("must3r")
        >>> model = ModelClass(config={"image_size": 512})
        >>> model.load()
        >>> result = model.reconstruct(video_input, output_dir)
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")

    module_path = _MODEL_REGISTRY[name]

    # Split into module and class name
    module_name, class_name = module_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except ImportError as e:
        logger.error(f"Failed to import model '{name}': {e}")
        raise ImportError(
            f"Could not import model '{name}'. "
            f"Make sure the required dependencies are installed. "
            f"Error: {e}"
        )
    except AttributeError as e:
        logger.error(f"Model class '{class_name}' not found in '{module_name}': {e}")
        raise


def list_models() -> list[str]:
    """
    List all available model names.

    Returns:
        List of model names that can be passed to get_model()
    """
    return list(_MODEL_REGISTRY.keys())


def register_model(name: str, module_path: str) -> None:
    """
    Register a new model at runtime.

    Args:
        name: Model name to register
        module_path: Full module path including class name
                    (e.g., "my_package.models.MyModel")

    Example:
        >>> register_model("my_model", "my_package.models.MyModel")
        >>> model = get_model("my_model")
    """
    if name in _MODEL_REGISTRY:
        logger.warning(f"Overwriting existing model registration for '{name}'")

    _MODEL_REGISTRY[name] = module_path
    logger.info(f"Registered model '{name}' -> {module_path}")


__all__ = [
    "BaseModel",
    "get_model",
    "list_models",
    "register_model",
]
