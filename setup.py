import re
from pathlib import Path
from setuptools import setup


def _read_pyproject_version():
    """Read version from pyproject.toml without importing any build tools."""
    pyproject = Path(__file__).parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find version in pyproject.toml")
    return m.group(1)


def _read_readme():
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


setup(
    name="unifiedefficientloader",
    version=_read_pyproject_version(),
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    packages=["unifiedefficientloader", "unifiedefficientloader.uel"],
)
