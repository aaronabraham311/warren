"""Entrypoint shim for the eval replay command.

    python -m agent.eval --golden-set --output runs/eval-2026-05-10.json

The implementation lives in ``eval/runner.py`` alongside the golden set and the fixtures
it replays; ``python -m eval.runner`` works identically.
"""

from eval.runner import main

if __name__ == "__main__":
    main()
