#!/usr/bin/env python3
"""Render the four-L4 README figures using the qualified drop-in measurements."""
import sys
from plot_nccl_dropin import main

if __name__ == '__main__':
    sys.argv += ['--topology', '4l4']
    main()
