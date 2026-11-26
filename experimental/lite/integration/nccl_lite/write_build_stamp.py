"""Track local adapter build inputs without shell-interpreting flag values."""
import json
import os
from pathlib import Path
import sys


def main():
    path = Path(sys.argv[1])
    fields = ['POLICY', 'NVCC', 'CXX', 'NVCC_FLAGS', 'INCLUDES', 'LIBS']
    data = {field.lower(): os.environ['UCCL_LOCAL_BUILD_' + field] for field in fields}
    text = json.dumps(data, sort_keys=True, indent=2) + '\n'
    if path.exists() and path.read_text() == text:
        return
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
