"""Run auditable agent-strategy ablations instead of selecting the first candidate."""

import sys

from omniintents.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ablate", *sys.argv[1:]]))
