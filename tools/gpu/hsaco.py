#!/usr/bin/env python3
"""Carve the embedded AMD code objects out of a cajeta exe and report each
kernel's descriptor: VGPRs, SGPRs, spills, scratch, LDS, occupancy.
usage: hsaco.py <exe> [kernel-substring] [--isa]"""
import sys, struct, subprocess, os, re
exe = sys.argv[1]; want = None; isa = False
for a in sys.argv[2:]:
    if a == "--isa": isa = True
    else: want = a
data = open(exe, 'rb').read()
out = os.environ.get("HSACO_OUT", "tmp/hsaco"); os.makedirs(out, exist_ok=True)
READELF = "/home/julian/.local/lib/rocm/llvm/bin/llvm-readelf"
OBJDUMP = "/home/julian/.local/lib/rocm/llvm/bin/llvm-objdump"
if not os.path.exists(READELF): READELF = "llvm-readelf"
i = 0; n = 0
while True:
    i = data.find(b"\x7fELF", i)
    if i < 0: break
    if data[i+4] == 2 and data[i+5] == 1 and struct.unpack_from("<H", data, i+18)[0] == 0xE0:
        shoff = struct.unpack_from("<Q", data, i+40)[0]
        shentsize, shnum = struct.unpack_from("<HH", data, i+58)
        size = shoff + shentsize * shnum
        blob = data[i:i+size]
        path = f"{out}/co{n}.hsaco"
        open(path, "wb").write(blob)
        notes = subprocess.run([READELF, "--notes", path], capture_output=True, text=True).stdout
        names = re.findall(r"\.name:\s+(\S+)", notes)
        kn = [x for x in names if not x.endswith(".kd")]
        for k in set(kn):
            if want and want not in k: continue
            # per-kernel block
            m = re.search(r"\.name:\s+" + re.escape(k) + r"\n(.*?)(?=\n\s+- \.|\Z)", notes, re.S)
            blk = notes[notes.find(k):]
            def g(key):
                mm = re.search(r"\." + key + r":\s+(\S+)", blk)
                return mm.group(1) if mm else "?"
            print(f"{k}: vgpr {g('vgpr_count')} agpr {g('agpr_count')} sgpr {g('sgpr_count')} "
                  f"vgpr_spill {g('vgpr_spill_count')} sgpr_spill {g('sgpr_spill_count')} "
                  f"scratch {g('private_segment_fixed_size')} lds {g('group_segment_fixed_size')} "
                  f"wavefront {g('wavefront_size')}  [{path}]")
            if isa:
                asm = subprocess.run([OBJDUMP, "-d", "--mcpu=gfx1151", path], capture_output=True, text=True).stdout
                open(f"{out}/{k}.s", "w").write(asm)
                body = asm
                loads = len(re.findall(r"\bglobal_load_\w+", body))
                ld = {}
                for mm in re.findall(r"\b(global_load_\w+|buffer_load_\w+|scratch_\w+|s_waitcnt\w*|v_dot4\w*|ds_\w+)", body):
                    ld[mm] = ld.get(mm, 0) + 1
                print("   ", ", ".join(f"{k2} {v}" for k2, v in sorted(ld.items())))
                print(f"    ISA at {out}/{k}.s")
        n += 1
        i += size
    else:
        i += 4
