python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install 'triton>=3.0.0'
pip install numpy
pip install pytest

cd ep/bench/baseline 
python setup_pack_unpack.py build_ext --inplace
python -m pytest test_pack_unpack_triton.py -v
python benchmark_pack_unpack.py