
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_git_info() -> Tuple[Optional[str], str]:
    """
    Get git commit hash and repository status.
    
    Returns:
        Tuple of (commit_hash, status) where status is "clean" or "dirty"
    """
    try:
        from git import Repo, InvalidGitRepositoryError
        
        # Try to find the git repository (start from current file's directory)
        current_path = Path(__file__).parent.parent.parent
        repo = None
        
        # Walk up the directory tree to find .git
        for path in [current_path] + list(current_path.parents):
            git_dir = path / ".git"
            if git_dir.exists():
                try:
                    repo = Repo(str(path))
                    break
                except InvalidGitRepositoryError:
                    continue
        
        if repo is None:
            return None, "unknown"
        
        # Get commit hash
        commit_hash = repo.head.commit.hexsha[:12]  # Short hash
        
        # Check if working tree is dirty
        is_dirty = repo.is_dirty()
        status = "dirty" if is_dirty else "clean"
        
        return commit_hash, status
        
    except ImportError:
        logger.warning("GitPython not available, skipping git info")
        return None, "unknown"
    except Exception as e:
        logger.warning(f"Failed to get git info: {e}")
        return None, "unknown"

