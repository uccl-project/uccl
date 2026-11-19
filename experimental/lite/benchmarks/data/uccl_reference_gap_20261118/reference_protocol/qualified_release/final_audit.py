import json,csv,hashlib,importlib.util,collections,re,shlex,time,os
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_default_release';measured=root/'build/uccl_rs16_graph512'
def require(v,msg):
 if not v:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
paired=module('paired_audit',root/'tests/nccl_mscclpp/audit_uccl_pair.py').audit(measured/'paired')
batch=module('batch_audit',root/'tests/nccl_mscclpp/audit_nccl_sweeps.py').audit(measured,root/'build/nccl-tests-mpi',root/'build/uccl_readme_reference/fresh')
require(paired==json.loads((measured/'paired/raw_log_audit.json').read_text()),'paired audit changed')
require(batch==json.loads((measured/'batch_raw_log_audit.json').read_text()),'batch audit changed')
require(paired['zero_slowdown_passed'] and paired['rows']==126 and paired['runs']==27,'paired zero-regression gate')
require(batch['slower_rows']==0 and batch['rows']==126 and batch['runs']==9,'batch gate')
refs={'ag':'3bb71d255b4edfeeddcd00a389fea88f9150f546327460c392e06c94d93cb658','ar':'a0edd320e96eda05417a49b34146d5c98f8c8a5c1130fedc67be93c3e9ef43ca','rs':'d01c6009e2d20fbeb109291b6188dd0181f10fbf2bb5e3b9d0bee5f4f0f18235'}
manifest=json.loads((measured/'paired/manifest.json').read_text());require(manifest['reference_hashes']==refs,'original references changed')
for path,cases in [(measured/'paired/checker_rejection_cases.json',{'nonfinite','missing_route','duplicate_sample','changed_mean'}),(measured/'batch_checker_rejection_cases.json',{'nonfinite','missing_route','duplicate_row','changed_mean'})]:
 data=json.loads(path.read_text());require(len(data)==4 and {x['case'] for x in data}==cases and all(x['rejected'] for x in data),'negative audit fixtures')
state=json.loads((base/'state.json').read_text());require(state['exit_code']==0 and state['default_policy']=='graph512' and state['abi_calls_per_rank']==234,'default build/ABI')
policy=json.loads((root/'build/uccl_policy_graph512/state.json').read_text());require(policy['exit_code']==0 and policy['runtime_identical_to_measured'] and policy['policy_switch_invalidates'] and policy['unchanged_policy_no_rebuild'],'explicit policy build')
library=root/state['library'];identity=module('elf_identity',root/'build/elf_runtime_identity.py');actual=identity.inspect(library);original=identity.inspect(measured/'nccl_lite/libuccl_lite_nccl_strict.so')
require(actual['runtime_sha256']==original['runtime_sha256']==state['runtime_sha256'],'loaded code/data differ')
require(actual['raw_sha256']==state['library_sha256'] and original['raw_sha256']==paired['candidate_sha256']==batch['candidate_sha256'],'artifact hashes')
require((root/'build/nccl_lite/build.policy').read_text().strip()=='graph512','default build stamp')
require('UCCL_LITE_NCCL_POLICY ?= graph512' in (root/'integration/nccl_lite/policy.mk').read_text(),'default policy')
require(not any(x.startswith('UCCL_LITE_NCCL_POLICY=') for x in json.loads((base/'build_command.json').read_text())),'default was overridden')
for alias in ['build/nccl_lite/strict/libnccl.so.2','build/nccl_lite/libuccl_lite_nccl_mscclpp.so']:require((root/alias).resolve()==library.resolve(),'loader alias')
dynamic=(base/'dynamic.log').read_text();require('Library soname: [libnccl.so.2]' in dynamic and not any('NEEDED' in line and 'libnccl' in line for line in dynamic.splitlines()),'native dependency or SONAME')
queue=json.loads((measured/'queue_check/result.json').read_text());require(queue['correct'] and queue['calls_per_rank']==48 and queue['library_sha256']==original['raw_sha256'],'queued graph correctness')
for name,digest in json.loads((measured/'sources.json').read_text()).items():require(sha(root/name)==digest,'source changed: '+name)
for host in ['host0','host1']:
 old=json.loads((measured/f'abi/{host}.json').read_text());new=json.loads((base/f'abi/{host}.json').read_text())
 require(all(old[k]==new[k] for k in ['hostname','versions','uuids']),'host version/placement changed')
 a=dict(old['details']);b=dict(new['details']);require(a.pop('library_sha256')==original['raw_sha256'] and b.pop('library_sha256')==actual['raw_sha256'],'host library identity');require(a==b,'host hardware/dependencies changed')
 require(sha(base/'abi/libuccl_lite_nccl_strict.so')==actual['raw_sha256'],'staged default library')
environment=json.loads((measured/'environment.json').read_text());require(environment['before']==environment['after']=={'local':'','remote':''},'profiling environment')
validation=module('strict_validation',root/'tests/nccl_mscclpp/run.py')
smoke=json.loads((base/'smoke_state.json').read_text());require(smoke['exit_code']==0 and len(smoke['runs'])==3,'default stock-test smoke')
for item in smoke['runs']:
 op=item['operation'];number={'ag':1,'ar':3,'rs':2}[op];log=(base/f'smoke_{op}.log').read_text();data=validation.checked_strict(log,number)
 require(item['state']=='complete' and item['exit_code']==0 and set(data['rows_us'])=={1<<p for p in range(10,31)},'default smoke coverage')
 require(all(all(__import__('math').isfinite(x) and x>0 for x in row) for row in data['rows_us'].values()),'default smoke timings')
 routes=collections.Counter(tuple(map(int,x)) for x in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',log));require(routes==collections.Counter((r,number,1<<p,84) for r in range(8) for p in range(10,31)),'default route coverage')
 require(sha(Path(item['command'][item['command'].index('--')+1]))==batch['nccl_tests_sha256'][op],'default test executable changed')
require('Strict ABI full-buffer checks passed: 234 calls/rank, two lifetimes' in (base/'abi/abi.log').read_text(),'default ABI marker')
rows=list(csv.DictReader((measured/'paired/comparison.csv').open()));worst={op:max(float(r['ratio']) for r in rows if r['operation']==op) for op in ['ag','ar','rs']}
# Capture every local CUDA dependency, plus shim/build/test inputs. External SDK
# dependencies are hashed, never vendored into the source snapshot.
logical=(root/'build/nccl_lite/mscclpp_backend.d').read_text().replace('\\\n',' ').splitlines()[0]
paths=[Path(x) for x in shlex.split(logical.split(':',1)[1])]
paths += [root/x for x in ['Makefile','src/cpu_router/module.mk','integration/nccl_lite/policy.mk','integration/nccl_lite/mscclpp_strict.cc','integration/nccl_lite/host_control.h','integration/nccl_lite/mscclpp_exports.version','integration/nccl_lite/mscclpp_profile.py','tests/nccl_mscclpp/run.py','tests/nccl_mscclpp/reference_timing.cu','tests/nccl_mscclpp/strict_abi.cu','tests/nccl_mscclpp/graph_queue.cu','tests/nccl_mscclpp/audit_uccl_pair.py','tests/nccl_mscclpp/audit_nccl_sweeps.py']]
dependencies={};local={}
for path in paths:
 path=path.resolve();require(path.is_file(),'missing source dependency: '+str(path));dependencies[str(path)]=sha(path)
 try:rel=path.relative_to(root)
 except ValueError:continue
 local[str(rel)]=dependencies[str(path)]
(base/'source_dependencies.json').write_text(json.dumps({'local':local,'all':dependencies},indent=2)+'\n')
report={'passed':True,'scope':'two recorded four-L4 hosts; FP32 AG and SUM AR/RS;21 powers of two1KiB–1GiB;both layouts;eager;oneGPU/process;recorded exact profile','paired':{'runs':27,'rows':126,'slower_rows':0,'worst_latency_ratio':worst},'nccl_tests':{'runs':9,'rows':126,'slower_rows':0,'binary_sha256':batch['nccl_tests_sha256']},'reference_sha256':refs,'default_policy':'graph512','measured_library_sha256':original['raw_sha256'],'default_library_sha256':actual['raw_sha256'],'loaded_runtime_sha256':actual['runtime_sha256'],'runtime_identity_scope':'all allocated ELF sections/layout andprogram headers,includingCUDAfatbins;onlyGNUbuild-ID descriptor masked;nonallocateddebug/symbolmetadata excluded','default_abi_calls_per_rank':234,'queued_graph_calls_per_rank':48,'default_stock_test_smokes':3,'native_nccl_loaded':False,'native_fallback':False,'audit_rejection_cases':8,'old_negative_results_retained':True,'completed_at':time.time()}
(base/'completion_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
