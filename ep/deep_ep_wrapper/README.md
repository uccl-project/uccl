## DeepEP Wrapper of UCCL-EP

First build and install UCCL-EP:
```bash
pushd ../../
bash build.sh cuda ep
uv pip install wheelhouse-cuda/uccl-*.whl
popd
```

Then install UCCL-EP's drop-in replacement for DeepEP:
```bash
python setup.py install
```
