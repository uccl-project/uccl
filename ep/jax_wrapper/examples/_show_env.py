import os
print("env:", os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT"), os.environ.get("RANK"), os.environ.get("LOCAL_RANK"), flush=True)
