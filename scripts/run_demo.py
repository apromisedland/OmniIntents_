"""Legacy module entry point; use the installed omniintents command."""

import sys

from omniintents.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
