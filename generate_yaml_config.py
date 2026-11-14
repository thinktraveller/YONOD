"""Generate or validate schema-2 YAML without launching modeling."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Interactive schema-2 YAML configuration generator. A CSV starts '
            'the shared preparation wizard and writes config/<task>.yaml; '
            'an existing YAML is validated only. This entrypoint never '
            'calculates descriptors, fits models, or predicts.'
        ),
        epilog=(
            'After generation, launch yonod.py separately and provide the '
            'saved YAML path at its first prompt. Preparation CSVs and input '
            'validation reports belong to the isolated result/<task>/ folder.'
        ),
    )
    parser.parse_args(argv)
    wizard = runpy.run_path(
        str(Path(__file__).resolve().with_name('yonod.py')),
        run_name='yonod_configuration_wizard',
    )
    try:
        return int(wizard['main'](generate_only=True) or 0)
    except (EOFError, OSError, ValueError) as exc:
        print(f'[error] YAML configuration generation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
