import sys
import os
import os.path as path


def add_path_to_dust3r(ckpt):
    here_path = os.path.dirname(os.path.abspath(ckpt))
    repo_src = os.path.join(os.path.dirname(__file__), "src")

    # Force the current repo src/ to win import resolution, even when the
    # checkpoint lives next to an older source tree with its own dust3r package.
    if repo_src in sys.path:
        sys.path.remove(repo_src)
    sys.path.insert(0, repo_src)

    # Keep compatibility with older checkpoints living next to source, but only
    # as a fallback after the current repo has already taken precedence.
    if here_path and here_path != repo_src:
        if here_path in sys.path:
            sys.path.remove(here_path)
        sys.path.append(here_path)
