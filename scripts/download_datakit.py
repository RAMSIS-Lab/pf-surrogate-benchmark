"""Download genco-pf-datakit base/seed0 checkpoints and the matching gridfm/pf_small_<case> datasets into cache/."""
import sys, os
from huggingface_hub import snapshot_download
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cases = sys.argv[1:]
for c in cases:
    snapshot_download("gridfm/genco-pf-datakit", allow_patterns=[f"{c}/base/seed0/**"], local_dir="cache/hf/genco-pf-datakit",
                      revision="f5eb20ee798b9847b723c91874af85d32a96a468")
    snapshot_download(f"gridfm/pf_small_{c}", repo_type="dataset", local_dir=f"cache/hf/pf_small_{c}", max_workers=16,
                      allow_patterns=["*.parquet/**", "README.md", "args.log"])
    print("done", c, flush=True)
