## DeepEP Wrapper of UCCL-EP

First build and install UCCL-EP:
```bash
cd uccl
PER_EXPERT_BATCHING=1 bash build.sh cu12 ep
uv pip install wheelhouse-cu12/uccl-*.whl
```

Then install UCCL-EP's drop-in replacement for DeepEP:
```bash
cd ep/deep_ep_wrapper
python setup.py install
```
