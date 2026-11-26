"""Build a NCCL-only copy of the existing checked FP32 AG/RS harness."""
from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
OUT = ROOT / "build/nccl_tuned_4l4_20261115"
OUT.mkdir(parents=True, exist_ok=True)
source = (ROOT / "src/collectives/peer_ag_rs.cu").read_text()
anchor = "        if (packetEligible) backends.push_back(Backend::PacketRouter);\n"
assert source.count(anchor) == 1
modified = source.replace(anchor, anchor + "        backends = {Backend::Nccl};  // Isolated NCCL tuning control.\n")
anchor = "          for (auto backend : order)\n"
assert modified.count(anchor) == 1
modified = modified.replace(anchor, "          order = backends;\n" + anchor)
(OUT / "peer_nccl.cu").write_text(modified)
(STUDY / "source.patch").write_text("".join(difflib.unified_diff(
    source.splitlines(keepends=True), modified.splitlines(keepends=True),
    fromfile="src/collectives/peer_ag_rs.cu",
    tofile="build/nccl_tuned_4l4_20261115/peer_nccl.cu")))
