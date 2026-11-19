import array,json,statistics,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('build/uccl_persistent_timeline/trace_1g')
names=['reuse_wait','upload_copy','upload_publish','local_wait','local_copy','remote_wait','remote_copy','consume_publish']
summary={};rank0=[]
for rank in range(8):
 words=array.array('Q');words.frombytes((root/f'rank_{rank}.bin').read_bytes())
 meta=json.loads((root/f'rank_{rank}.json').read_text());blocks=words[0]&0xffffffff;assert words[0]>>32==1 and blocks==meta['blocks']
 pieces=(meta['shard_bytes']+meta['piece_bytes']-1)//meta['piece_bytes'];origin=words[1]>>32
 phases=collections.defaultdict(list);byblock={};ends=collections.defaultdict(dict);copyspans=collections.defaultdict(dict);pairs=[]
 for b in range(blocks):
  begin=None;counts=collections.Counter();totals=collections.defaultdict(float)
  for w in words[1+b::blocks]:
   if not w:continue
   tag=w&0xffffffff;t=w>>32;event=(tag>>2)&1023;kind=tag&3;assert (tag>>12)&4095==b and event<8
   if kind==0:assert begin is None;begin=(event,t)
   else:
    assert kind==1 and begin is not None and begin[0]==event
    d=((t-begin[1])&0xffffffff)/1000;s=((begin[1]-origin+(1<<31))%(1<<32)-(1<<31))/1000
    piece=counts[event];counts[event]+=1;phases[event].append(d);totals[names[event]]+=d
    pairs.append((b,event,piece,s,d))
    if event==7:ends[piece][b]=s+d
    if event in [4,6]:copyspans[event,piece][b]=(s,d)
    begin=None
  expected=range(3) if b<meta['upload_blocks'] else range(3,8)
  assert all(counts[e]==pieces for e in expected),(rank,b,counts,pieces)
  byblock[b]={k:v/pieces for k,v in totals.items()}
 last=collections.Counter(max(v,key=v.get) for v in ends.values())
 skew={}
 for e in [4,6]:
  walls=[];slow=[]
  for p in range(pieces):
   v=copyspans[e,p];walls.append(max(s+d for s,d in v.values())-min(s for s,d in v.values()));slow.append(max(d for s,d in v.values()))
  skew[names[e]]={'mean_wall_us':statistics.mean(walls),'mean_slowest_block_us':statistics.mean(slow)}
 summary[rank]={'metadata':meta,'phase_us':{names[e]:{'mean':statistics.mean(v),'median':statistics.median(v),'max':max(v)} for e,v in phases.items()},'block_mean_us':byblock,'last_consumer_counts':dict(last),'copy_phase_skew':skew}
 if rank==0:rank0=pairs
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
colors=['#aaaaaa','#17becf','#9467bd','#cccccc','#1f77b4','#dddddd','#ff7f0e','#2ca02c']
fig,ax=plt.subplots(figsize=(13,8))
for b in range(summary[0]['metadata']['blocks']):
 for e in range(8):
  spans=[(s,d) for bb,ee,p,s,d in rank0 if bb==b and ee==e and p<8]
  if spans:ax.broken_barh(spans,(b-.4,.8),facecolors=colors[e])
ax.set_yticks(range(32),[f'P{b}' if b<4 else f'C{b-4}' for b in range(32)],fontsize=8)
ax.invert_yaxis();ax.set_xlabel('GPU globaltimer relative to block 0 (microseconds)');ax.set_ylabel('Block role and index')
ax.set_title('Persistent host AG, rank 0, first 8 packets (instrumented, not scored)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c,label=n) for c,n in zip(colors,names)],ncol=4,loc='upper center',bbox_to_anchor=(.5,-.09),fontsize=8)
fig.tight_layout();fig.savefig(root/'rank0_first8.png',dpi=150);plt.close(fig)
for r,v in summary.items():print(r,{k:round(p['mean'],2) for k,p in v['phase_us'].items()},'last',v['last_consumer_counts'])
