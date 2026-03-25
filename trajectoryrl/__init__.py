"""TrajectoryRL - Bittensor subnet for optimizing AI agent policies."""
from pathlib import Path

_version_file = Path(__file__).resolve().parent.parent / "VERSION"
if _version_file.is_file():
    __version__ = _version_file.read_text().strip()
else:
    from importlib.metadata import version
    __version__ = version("trajectoryrl")

from trajectoryrl.protocol import PackRequest, PackResponse
from trajectoryrl.utils import validate_opp_schema
__all__ = ["PackRequest", "PackResponse", "validate_opp_schema"]
