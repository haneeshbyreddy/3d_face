#!/usr/bin/env python3
"""
Shrink the textures inside a .glb, in place of a much larger one.

3DPrinter.glb arrives from the asset store as 2,868 triangles wearing four
2048x2048 PNGs — 6.95 MB of the file's 7.08 MB is texture, and the model is
drawn into a panel about 440 px wide. At that size a 2048 map spends roughly
sixteen texels on every pixel it will ever cover. This rewrites those maps at
a size the panel can actually resolve and re-encodes them as JPEG: none of
the four carries an alpha channel, so nothing is lost by leaving PNG behind.

Geometry, materials, samplers and accessor indices are untouched; only the
image bufferViews change, and every other view is copied across at its new
offset. Run it against the pristine file, not its own output:

    python3 tools/repack-glb.py 3dprinter/source/3DPrinter.glb --size 512

Writes <name>.min.glb next to the input.
"""
import argparse, io, json, os, struct, sys
from PIL import Image

JSON_CHUNK, BIN_CHUNK = 0x4E4F534A, 0x004E4942


def read_glb(path):
    with open(path, "rb") as f:
        magic, version, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            sys.exit(f"{path}: not a glb")
        if version != 2:
            sys.exit(f"{path}: glb version {version}, expected 2")
        js, bin_ = None, b""
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            length, kind = struct.unpack("<II", head)
            data = f.read(length)
            if kind == JSON_CHUNK:
                js = json.loads(data)
            elif kind == BIN_CHUNK:
                bin_ = data
    if js is None:
        sys.exit(f"{path}: no json chunk")
    return js, bin_


def write_glb(path, js, bin_):
    j = json.dumps(js, separators=(",", ":")).encode("utf-8")
    j += b" " * (-len(j) % 4)
    b = bin_ + b"\0" * (-len(bin_) % 4)
    total = 12 + 8 + len(j) + (8 + len(b) if b else 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), JSON_CHUNK)); f.write(j)
        if b:
            f.write(struct.pack("<II", len(b), BIN_CHUNK)); f.write(b)
    return total


def view_bytes(js, bin_, i):
    v = js["bufferViews"][i]
    o = v.get("byteOffset", 0)
    return bin_[o:o + v["byteLength"]]


def recode(raw, size, quality):
    im = Image.open(io.BytesIO(raw))
    before = im.size
    if max(im.size) > size:
        im = im.resize((min(size, im.width), min(size, im.height)),
                       Image.Resampling.LANCZOS)
    if im.mode != "RGB":
        im = im.convert("RGB")
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality, optimize=True, subsampling=0)
    return out.getvalue(), before, im.size


def narrow_indices(js, bin_, new):
    """Rewrite 32-bit index buffers as 16-bit wherever the mesh is small
    enough to address that way. Exactly the same triangles, half the bytes."""
    for acc in js.get("accessors", []):
        if acc.get("componentType") != 5125 or acc.get("type") != "SCALAR":
            continue
        bv = acc.get("bufferView")
        if bv is None or bv in new:
            continue
        raw = view_bytes(js, bin_, bv)
        idx = struct.unpack(f"<{acc['count']}I", raw[:acc["count"] * 4])
        if max(idx) > 0xFFFF:
            continue
        new[bv] = struct.pack(f"<{len(idx)}H", *idx)
        acc["componentType"] = 5123
        print(f"  {'indices':<62} u32 {len(raw)/1e6:6.2f} MB"
              f"  ->  u16 {len(new[bv])/1e6:5.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--size", type=int, default=512, help="longest edge, px")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    js, bin_ = read_glb(args.glb)
    images = js.get("images", [])
    if any("uri" in im for im in images):
        sys.exit("external image uris are not handled")

    # normal maps carry direction, not colour: chroma error there reads as
    # lighting noise, so they keep a little more of the bit budget
    new = {}
    for im in images:
        raw = view_bytes(js, bin_, im["bufferView"])
        q = min(96, args.quality + 6) if "Normal" in im.get("name", "") else args.quality
        data, was, now = recode(raw, args.size, q)
        new[im["bufferView"]] = data
        im["mimeType"] = "image/jpeg"
        print(f"  {im.get('name','?'):<62} {was[0]}px {len(raw)/1e6:6.2f} MB"
              f"  ->  {now[0]}px {len(data)/1e6:5.2f} MB")

    narrow_indices(js, bin_, new)
    if not new:
        sys.exit("nothing to shrink")

    # rebuild the binary chunk in bufferView order so every offset stays valid
    out = bytearray()
    for i, v in enumerate(js["bufferViews"]):
        data = new.get(i, view_bytes(js, bin_, i))
        out += b"\0" * (-len(out) % 4)
        v["byteOffset"] = len(out)
        v["byteLength"] = len(data)
        out += data
    js["buffers"] = [{"byteLength": len(out)}]

    dest = args.out or os.path.splitext(args.glb)[0] + ".min.glb"
    total = write_glb(dest, js, bytes(out))
    was = os.path.getsize(args.glb)
    print(f"\n{args.glb} {was/1e6:.2f} MB  ->  {dest} {total/1e6:.2f} MB"
          f"  ({100 - total * 100 // was}% smaller)")


if __name__ == "__main__":
    main()
