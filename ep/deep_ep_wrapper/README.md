## DeepEP Wrapper of UCCL-EP

First build and install UCCL-EP:
```bash
cd uccl/ep
bash build.sh cuda --install
```

Then install UCCL-EP's drop-in replacement for DeepEP:
```bash
cd deep_ep_wrapper
python setup.py install
```
