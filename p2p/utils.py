import sys
import subprocess
import resource


def set_files_limit():
    """
    Configure files limit for high-performance communication.
    """
    print("Setting up files limit...", file=sys.stderr)

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current fd limit: soft={soft}, hard={hard}", file=sys.stderr)

        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"fd limit raised to: {hard}", file=sys.stderr)
        else:
            print("fd limit already at maximum", file=sys.stderr)

    except Exception as e:
        print(f"Failed to set fd limit: {e}", file=sys.stderr)
