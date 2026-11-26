"""Compare ELF64 loadable bytes/layout; mask only GNU build-ID descriptor data.

Non-allocated symbol/debug tables can contain NVCC temporary filenames. Dynamic
symbols, relocations, code, constants, CUDA fatbins and program headers remain in
the identity. This is build provenance, not a performance or correctness gate.
"""
import hashlib,json,struct,sys
from pathlib import Path

def inspect_bytes(blob):
 assert blob[:6]==b'\x7fELF\x02\x01','requires ELF64 little-endian'
 h=struct.unpack_from('<16sHHIQQQIHHHHHH',blob)
 ident,typ,machine,version,entry,phoff,shoff,flags,ehsize,phentsize,phnum,shentsize,shnum,shstrndx=h
 assert ehsize==64 and phentsize==56 and shentsize==64
 assert version==1 and shnum>0 and shstrndx<shnum
 headers=[struct.unpack_from('<IIQQQQIIQQ',blob,shoff+i*shentsize) for i in range(shnum)]
 strings=headers[shstrndx];names=blob[strings[4]:strings[4]+strings[5]]
 sections=[];allocated=[]
 for header in headers:
  name,type_,section_flags,address,offset,size,link,info,align,entsize=header
  name=names[name:names.index(b'\0',name)].decode()
  data=b'' if type_==8 else blob[offset:offset+size]
  assert type_==8 or len(data)==size
  sections.append({'name':name,'offset':offset,'size':size,'allocated':bool(section_flags&2)})
  if not section_flags&2:continue
  if name=='.note.gnu.build-id':
   assert type_==7 and len(data)>=16
   namesz,descsz,note_type=struct.unpack_from('<III',data)
   start=12+((namesz+3)//4)*4
   assert namesz==4 and data[12:16]==b'GNU\0' and note_type==3 and start+descsz<=len(data)
   data=data[:start]+bytes(descsz)+data[start+descsz:]
  allocated.append({'name':name,'type':type_,'flags':section_flags,'address':address,'offset':offset,'size':size,'alignment':align,'entry_size':entsize,'sha256':hashlib.sha256(data).hexdigest()})
 assert any(s['name']=='.text' for s in allocated)
 for section in sections:
  if 'fatbin' in section['name'].lower():assert section['allocated'],section
 identity={'elf_ident':ident.hex(),'type':typ,'machine':machine,'version':version,'entry':entry,'flags':flags,'program_headers':[list(struct.unpack_from('<IIQQQQQQ',blob,phoff+i*phentsize)) for i in range(phnum)],'allocated_sections':allocated}
 digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return {'runtime_sha256':digest,'raw_sha256':hashlib.sha256(blob).hexdigest(),'identity':identity,'sections':sections}

def inspect(path):return inspect_bytes(Path(path).read_bytes())

if __name__=='__main__':
 result=inspect(sys.argv[1])
 if len(sys.argv)>2:
  other=inspect(sys.argv[2]);print(json.dumps({'runtime_equal':result['runtime_sha256']==other['runtime_sha256'],'first_raw':result['raw_sha256'],'second_raw':other['raw_sha256'],'first_runtime':result['runtime_sha256'],'second_runtime':other['runtime_sha256']},indent=2));sys.exit(0 if result['runtime_sha256']==other['runtime_sha256'] else 2)
 print(json.dumps(result,indent=2))
