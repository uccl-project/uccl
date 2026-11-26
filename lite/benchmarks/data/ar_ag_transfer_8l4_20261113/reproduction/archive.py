import datetime,gzip,hashlib,json,re,shutil,subprocess
from pathlib import Path
LITE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=LITE/'benchmarks/data/ar_ag_transfer_8l4_20261113'
for directory in ['raw','validation','reproduction']:(DATA/directory).mkdir(parents=True,exist_ok=True)
for path in sorted(OUT.iterdir()):
 if path.is_file() and path.name.startswith(('screen_','new_','final_','auto_')) and path.suffix in ('.csv','.json','.profile','.log'):
  target=DATA/'raw'/path.name
  if path.name.endswith('_samples.csv'):
   target=target.with_suffix('.csv.gz');target.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
  else:shutil.copyfile(path,target)
for name in ['run_sweeps.py','run_screen.py','run_existing_extra.py','run_new_screen.py','run_final.py','run_correctness.py','run_auto.py','run_ar_cpu_controls.py','archive.py']:
 shutil.copyfile(OUT/name,DATA/'reproduction'/name)
for name in ['source_hashes.json','implementation.patch','correctness.json','auto_replay.json']:
 shutil.copyfile(OUT/name,DATA/name)
for record in json.loads((OUT/'correctness.json').read_text()):
 shutil.copyfile(OUT/(record['name']+'.log'),DATA/'validation'/(record['name']+'.log'))
for name in ['build.log','build_final.log','correctness.log','auto.log']:
 shutil.copyfile(OUT/name,DATA/'validation'/name)
shutil.copyfile(LITE/'benchmarks/data/rs_submit_8l4_20261113/source_hashes.json',DATA/'source_hashes_before.json')
def binary(name):
 path=OUT/name
 elf=subprocess.check_output(['readelf','-n',str(path)],text=True)
 return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),elf_build_id=re.search(r'Build ID: (\w+)',elf).group(1))
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,driver_version','--format=csv'],text=True)
(DATA/'hardware.csv').write_text(hardware)
provenance={
 'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'hardware':{'gpu':'8 x NVIDIA L4 (SM89), no cross-NUMA peer access','ranks_by_numa':[[0,1,2,3],[4,5,6,7]],'cpu':'Intel(R) Xeon(R) CPU @ 2.20GHz; 96 logical CPUs, two NUMA nodes','main_cpu':47,'cuda':'12.9','driver':'580.178.04','nccl':'2.28.9'},
 'method':{'dtype':'BF16','allreduce':'in place','allgather':'out of place','bytes':'complete tensor bytes per rank: AR input / AG output','range':[1024,4194304],'warmups':10,'paired_repeats':5,'checked_calls_per_repeat_per_backend':20,'confirmation_sweeps':2,'order':'a: AR then AG, previous/batch/progress/combined/compact_combined; b: reverse operations and variants','point':'median of two sweep medians','sweep_range':'min/max of sweep medians, not confidence intervals','additional_controls':'After the main confirmation, AR CPU-parallel and CPU-parallel-compact run in two reversed-order sweeps without GPU progress. They are separate controls, not extra samples for earlier points.','per_size_best_variant':False,'timer':'original harness slowest-rank CUDA-event interval; includes host submission gaps; dense eager calls','baseline':'each variant has its own alternating paired NCCL control; P2P default on within NUMA, SHM across NUMA','exclusivity':'compute process list checked before and after each sweep; busy GPU aborts','screening':'preliminary out-of-place AR/AG existing-option sweeps; new_* uses in-place AR, out-of-place AG. Some screening overlapped CPU compilation. Confirmation runs after build/correctness completion.'},
 'variants':{'previous':'frozen pre-change binary, existing benchmark defaults (AR compact 4-worker/16-block rule at 128/256 KiB)','batch':'current binary plus --batch-cpu-jobs (all layouts)','progress':'batch plus --small-ar-gpu-progress-max-bytes 8192 --small-ag-gpu-progress-max-bytes 1048576 (per-source AG)','combined':'progress plus --gpu-launch-workers-per-node 4','compact_combined':'combined plus --workers-per-node 4 --transport-blocks 16 across all sizes','cpu_parallel':'batch plus --gpu-launch-workers-per-node 4, with CPU progress at every AR size','cpu_parallel_compact':'cpu_parallel plus --workers-per-node 4 --transport-blocks 16 across all sizes'},
 'primary_comparison':{'allreduce':'compact_combined','allgather':'combined','selection':'one fixed configuration per operation for the entire sweep; router timing and regressions inform this choice, not paired NCCL values; no per-size best variant selection'},
 'software':{'baseline_source_commit':'0dc35a3b1f722f0525a808239b12cfe3b6ff5f0e','baseline_source_hashes':'source_hashes_before.json (37 files verified against baseline commit)','current_source_hashes':'source_hashes.json','change':'implementation.patch','build_command':'make -C experimental/lite -j2 adaptive-collectives router-checks BUILD=build/ar_ag_transfer_20261113 CUDA_HOME=/usr/local/cuda-12.9 CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 NCCL_HOME=build/nccl-2.28.9'},
 'binaries':{'before':binary('collectives_before'),'current':binary('uccl_lite_collectives')},
 'validation':{'manifest':'correctness.json','checks':13,'queued_eager_cases':2928,'captured_mixed_calls':2000,'randomized_oracle_scalar_guard_checks':53138272,'calibration_replay':'auto_replay.json: matching known sizes follow eligibility; exact 1536-byte and changed-threshold keys retain fallback without router initialization'},
 'limits':['Options are off by default and need fresh exact local calibration.','Parallel eager submission reserves eight extra CPU threads and spins for up to 2 ms; call spacing and affinity matter. New AR/AG sparse-call performance is not measured.','GPU AR arithmetic rounds once in FP32, whereas the CPU path rounds intermediate BF16 NUMA partials.','No additional 2/4-MiB improvement established. No measurements above 4 MiB.','C API and multiprocess drop-in expose no new controls; no application speedup claimed.','Historical overview figures remain unchanged.'],
 'reproduction':'Copy reproduction scripts into experimental/lite/build/ar_ag_transfer_20261113 before execution, rebuild the current binary there, and reconstruct the baseline from the recorded commit in a separate checkout/build before copying it to collectives_before. Choose fresh output locations/names. Exact commands and clean environments are recorded in raw/*.json. Raw sample CSVs are losslessly gzip compressed.'
}
(DATA/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Archived measurements and validation in',DATA)
