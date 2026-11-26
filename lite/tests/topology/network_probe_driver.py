"""Exercise the monitor's real NVML exit race without weakening foreign-job checks."""
import importlib.util
from pathlib import Path
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('network_driver', root / 'benchmarks/probes/run_network.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
owned = {}
binary = '/experiment/network_probe'
def record(pid, name):
    return [dict(host='host0', apps=[['gpu0', pid, name, '320']])]
assert not driver.foreign_apps(record('123', binary), binary, owned)
assert not driver.foreign_apps(record('123', '[No data]'), binary, owned)
assert driver.foreign_apps(record('124', '[No data]'), binary, owned)
assert driver.foreign_apps(record('123', '/other/job'), binary, owned)
assert driver.foreign_apps(record('124', '/other/job'), binary, owned)
assert driver.idle(dict(apps=[], devices=[['0', 'gpu0', '0', '0']]), 1)
assert not driver.idle(dict(apps=[], devices=[['0', 'gpu0', '0', '320']]), 1)
assert not driver.idle(dict(apps=[], devices=[['0', 'gpu0', '100', '0']]), 1)
print('PASS: known probe PID cleanup, foreign PID/executable detection and idle gate')
