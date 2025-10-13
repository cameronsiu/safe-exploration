from pathlib import Path
import inspect

def get_project_root_dir() -> Path:
    """Return the absolute path to the project root."""
    # Get the file path of the caller
    current_file = Path(inspect.getfile(inspect.currentframe().f_back)).resolve()
    # Move up two levels: src/safe_exploration -> src -> project root
    return current_file.parents[3]

def get_config_path(filename: str = "defaults.yml") -> Path:
    """Return the absolute path to the config file."""
    root = get_project_root_dir()
    return root / "config" / filename