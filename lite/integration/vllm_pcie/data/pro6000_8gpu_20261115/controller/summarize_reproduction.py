"""Summarize fresh fine serving results; never reads committed historical data."""
import argparse,csv,json,math
from pathlib import Path

def gm(values):
    assert values and all(math.isfinite(x) and x>0 for x in values)
    return math.exp(sum(math.log(x) for x in values)/len(values))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    rows=[]
    for bs in (1,2,4,8,16,32,64):
        pair=[]
        for backend in ('nccl','pcieccl'):
            path=a.root/backend/'cases'/f'isl8192-osl1024-bs{bs}.json'
            d=json.loads(path.read_text());m=d['metrics']
            assert d['status']=='complete' and m['request_errors']==0
            assert d['case']['warmup_seconds']==60 and d['case']['measurement_seconds']==180
            assert m['requests_completed']>0
            pair.append(m)
        n,u=pair
        rows.append(dict(concurrency=bs,nccl_tok_s=n['output_token_throughput'],pcieccl_tok_s=u['output_token_throughput'],throughput_speedup=u['output_token_throughput']/n['output_token_throughput'],nccl_tpot_ms=n['tpot_ms']['p50'],pcieccl_tpot_ms=u['tpot_ms']['p50'],tpot_speedup=n['tpot_ms']['p50']/u['tpot_ms']['p50'],nccl_ttft_ms=n['ttft_ms']['p50'],pcieccl_ttft_ms=u['ttft_ms']['p50'],ttft_speedup=n['ttft_ms']['p50']/u['ttft_ms']['p50']))
    quality={b:json.loads((a.root/'quality'/f'gsm8k-{b}.json').read_text()) for b in ('nccl','pcieccl')}
    result=dict(rows=rows,geomean={k:gm([r[k] for r in rows]) for k in ('throughput_speedup','tpot_speedup','ttft_speedup')},quality=quality,caveats=['Single fine run per backend; no confidence interval','NCCL baseline disables vLLM custom and symmetric-memory all-reduce','Requested naive all2all maps to allgather_reducescatter in pinned vLLM','Concurrency is request concurrency, not fixed per-step token batch','70% constructed shared prefix is not measured cache hit rate'])
    (a.root/'comparison.json').write_text(json.dumps(result,indent=2))
    with (a.root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print('Concurrency  NCCL tok/s PCIeCCL tok/s  speedup  NCCL TPOT  PCIeCCL TPOT')
    for r in rows:print(f"{r['concurrency']:>11} {r['nccl_tok_s']:>11.2f} {r['pcieccl_tok_s']:>13.2f} {r['throughput_speedup']:>7.3f}x {r['nccl_tpot_ms']:>10.3f} {r['pcieccl_tpot_ms']:>13.3f}")
    print(json.dumps(result['geomean']))
    for b,q in quality.items():print(b,'GSM8K',q['accuracy'],'invalid',q['invalid_rate'])

if __name__=='__main__':main()
