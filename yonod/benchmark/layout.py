"""Output-directory contract for strict benchmark runs.

New runs keep machine-readable artefacts under ``docs/``, rendered images
under ``pictures/``, and human-readable reports under ``report/``.  The
resolver deliberately recognises the pre-contract layout so report rebuilds
and interrupted-run recovery continue to work for already-created runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkOutputLayout:
    """Resolved paths for one benchmark run.

    ``legacy`` is true only when a run already contains the old root-level
    manifest.  Fresh run directories always use the new layout; no existing
    files are moved or overwritten merely by resolving their paths.
    """

    run_dir: Path
    legacy: bool

    @property
    def docs(self) -> Path:
        return self.run_dir if self.legacy else self.run_dir / "docs"

    @property
    def manifests(self) -> Path:
        return self.docs / "manifests"

    @property
    def predictions(self) -> Path:
        return self.docs / "predictions"

    @property
    def folds(self) -> Path:
        return self.docs / "folds"

    @property
    def metrics(self) -> Path:
        return self.docs / "metrics"

    @property
    def state(self) -> Path:
        return self.docs / "state"

    @property
    def descriptors(self) -> Path:
        """Persistent descriptor artifacts consumed by every model task."""
        return self.run_dir / "descriptors"

    @property
    def pictures(self) -> Path:
        return self.run_dir / ("figures" if self.legacy else "pictures")

    @property
    def report(self) -> Path:
        return self.run_dir / ("reports" if self.legacy else "report")

    @property
    def picture_relative_to_report(self) -> str:
        return "../{0}".format(self.pictures.name)

    def artifact_reference(self, name: str) -> str:
        """Return a human-readable relative location for one artefact class."""
        return name if self.legacy else "docs/{0}".format(name)


def resolve_benchmark_output_layout(run_dir: Path | str) -> BenchmarkOutputLayout:
    """Return the layout for a new or previously-created benchmark run.

    A legacy manifest is the only signal that opts into old paths.  This
    prevents a pre-existing unrelated ``docs/`` directory from changing a
    historical run, while new runs consistently choose the three-directory
    contract.
    """

    root = Path(run_dir)
    legacy_manifest = root / "manifests" / "run_manifest.json"
    return BenchmarkOutputLayout(root, legacy=legacy_manifest.is_file())
