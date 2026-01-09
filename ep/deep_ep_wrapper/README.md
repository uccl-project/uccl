## DeepEP Wrapper of UCCL-EP

First build and install UCCL-EP:
```bash
cd uccl
bash build.sh cuda ep
uv pip install wheelhouse-cuda/uccl-*.whl
```

Then install UCCL-EP's drop-in replacement for DeepEP:
```bash
cd ep/deep_ep_wrapper
python setup.py install
```
