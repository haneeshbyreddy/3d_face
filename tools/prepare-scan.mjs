/* Prepare the scanned head for the web.
 *
 *   node tools/prepare-scan.mjs head.glb [targetTriangles]
 *
 * Reads the full-resolution scan and writes two things beside it:
 *   head.min.glb   — the same head, decimated by quadric error metric
 *   scan-profile   — the base64 silhouette table index.html draws the SVG busts from
 *
 * No dependencies. Node 22.
 */
import fs from "node:fs";
import path from "node:path";

const SRC = process.argv[2] || "head.glb";
const TARGET = Number(process.argv[3] || 70000);
const KEEP_SMALL = 8000;   // components under this many triangles pass through whole
const log = (...a) => console.log(...a);

/* ── read the GLB ─────────────────────────────────────────────────────── */
function readGLB(file) {
  const buf = fs.readFileSync(file);
  const len = buf.readUInt32LE(8);
  let off = 12, json = null, bin = null;
  while (off < len) {
    const clen = buf.readUInt32LE(off), ctype = buf.readUInt32LE(off + 4);
    const data = buf.subarray(off + 8, off + 8 + clen);
    if (ctype === 0x4e4f534a) json = JSON.parse(data.toString("utf8"));
    if (ctype === 0x004e4942) bin = data;
    off += 8 + clen;
    if (clen % 4) off += 4 - (clen % 4);
  }
  return { json, bin };
}

const CT = { 5120: [Int8Array, 1], 5121: [Uint8Array, 1], 5122: [Int16Array, 2],
             5123: [Uint16Array, 2], 5125: [Uint32Array, 4], 5126: [Float32Array, 4] };
const NC = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

const { json: G, bin: BIN } = readGLB(SRC);
function acc(ai) {
  const a = G.accessors[ai], bv = G.bufferViews[a.bufferView];
  const [Arr, sz] = CT[a.componentType], n = NC[a.type];
  const start = (bv.byteOffset || 0) + (a.byteOffset || 0);
  const stride = bv.byteStride || sz * n;
  if (stride === sz * n) return new Arr(BIN.buffer, BIN.byteOffset + start, a.count * n);
  const out = new Arr(a.count * n);
  for (let i = 0; i < a.count; i++)
    out.set(new Arr(BIN.buffer, BIN.byteOffset + start + i * stride, n), i * n);
  return out;
}

/* world matrices, walking the scene graph */
function mul(a, b) {                    // column-major 4x4
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
const IDENT = new Float64Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
function localMatrix(n) {
  if (n.matrix) return new Float64Array(n.matrix);
  const m = new Float64Array(IDENT);
  if (n.rotation) {
    const [x, y, z, w] = n.rotation;
    m[0] = 1-2*(y*y+z*z); m[1] = 2*(x*y+z*w);   m[2] = 2*(x*z-y*w);
    m[4] = 2*(x*y-z*w);   m[5] = 1-2*(x*x+z*z); m[6] = 2*(y*z+x*w);
    m[8] = 2*(x*z+y*w);   m[9] = 2*(y*z-x*w);   m[10] = 1-2*(x*x+y*y);
  }
  if (n.scale) for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[c*4+r] *= n.scale[c];
  if (n.translation) { m[12] = n.translation[0]; m[13] = n.translation[1]; m[14] = n.translation[2]; }
  return m;
}
const meshes = [];
(function walk(ni, parent) {
  const n = G.nodes[ni], m = mul(parent, localMatrix(n));
  if (n.mesh !== undefined) meshes.push({ mesh: n.mesh, m });
  (n.children || []).forEach((c) => walk(c, m));
})((G.scenes[G.scene || 0].nodes)[0], IDENT);

/* ── gather geometry, world space, one flat pool ───────────────────────── */
const groups = [];
for (const { mesh, m } of meshes) {
  for (const prim of G.meshes[mesh].primitives) {
    const p = acc(prim.attributes.POSITION);
    const idx = prim.indices !== undefined ? acc(prim.indices) : null;
    const nv = p.length / 3;
    const pos = new Float64Array(nv * 3);
    for (let i = 0; i < nv; i++) {
      const x = p[i*3], y = p[i*3+1], z = p[i*3+2];
      pos[i*3]   = m[0]*x + m[4]*y + m[8]*z  + m[12];
      pos[i*3+1] = m[1]*x + m[5]*y + m[9]*z  + m[13];
      pos[i*3+2] = m[2]*x + m[6]*y + m[10]*z + m[14];
    }
    const ind = idx ? Int32Array.from(idx) : Int32Array.from({ length: nv }, (_, i) => i);
    groups.push({ pos, ind, tris: ind.length / 3, name: G.meshes[mesh].name });
  }
}
log(`read ${groups.length} primitives, ${groups.reduce((s, g) => s + g.tris, 0).toLocaleString()} triangles`);

/* small primitives (the two eyes) are cheap and visually load-bearing: keep whole */
const keep = groups.filter((g) => g.tris < KEEP_SMALL);
const bulk = groups.filter((g) => g.tris >= KEEP_SMALL);
log(`keeping ${keep.length} small primitives whole (${keep.reduce((s,g)=>s+g.tris,0)} tris), ` +
    `decimating ${bulk.reduce((s,g)=>s+g.tris,0).toLocaleString()}`);

/* ── weld: the chunks are split at 65k vertices, not along seams ───────── */
function weld(parts) {
  const map = new Map();
  const px = [], py = [], pz = [];
  const ind = [];
  const Q = 1e5;
  for (const g of parts) {
    const remap = new Int32Array(g.pos.length / 3);
    for (let i = 0; i < remap.length; i++) {
      const k = `${Math.round(g.pos[i*3]*Q)},${Math.round(g.pos[i*3+1]*Q)},${Math.round(g.pos[i*3+2]*Q)}`;
      let v = map.get(k);
      if (v === undefined) {
        v = px.length; px.push(g.pos[i*3]); py.push(g.pos[i*3+1]); pz.push(g.pos[i*3+2]);
        map.set(k, v);
      }
      remap[i] = v;
    }
    for (let i = 0; i < g.ind.length; i += 3) {
      const a = remap[g.ind[i]], b = remap[g.ind[i+1]], c = remap[g.ind[i+2]];
      if (a !== b && b !== c && a !== c) ind.push(a, b, c);
    }
  }
  const pos = new Float64Array(px.length * 3);
  for (let i = 0; i < px.length; i++) { pos[i*3] = px[i]; pos[i*3+1] = py[i]; pos[i*3+2] = pz[i]; }
  return { pos, ind: Int32Array.from(ind) };
}

let t0 = Date.now();
const W = weld(bulk);
log(`welded: ${(W.pos.length/3).toLocaleString()} vertices, ${(W.ind.length/3).toLocaleString()} triangles ` +
    `(${Date.now()-t0} ms)`);

/* ── quadric error metric decimation (Garland & Heckbert) ──────────────── */
function decimate(pos, ind, targetTris) {
  const nv = pos.length / 3;
  let nf = ind.length / 3;
  const F = Int32Array.from(ind);
  const fAlive = new Uint8Array(nf).fill(1);
  const vAlive = new Uint8Array(nv).fill(1);
  const stamp = new Int32Array(nv);
  const Q = new Float64Array(nv * 10);

  const vf = new Array(nv);
  for (let i = 0; i < nv; i++) vf[i] = [];
  for (let f = 0; f < nf; f++)
    for (let k = 0; k < 3; k++) vf[F[f*3+k]].push(f);

  /* face plane, and its area — quadrics weighted by area so a dense flat
     region does not outvote a sparse curved one */
  const fn = new Float64Array(nf * 4);
  function facePlane(f) {
    const a = F[f*3], b = F[f*3+1], c = F[f*3+2];
    const ax=pos[a*3], ay=pos[a*3+1], az=pos[a*3+2];
    const ux=pos[b*3]-ax, uy=pos[b*3+1]-ay, uz=pos[b*3+2]-az;
    const vx=pos[c*3]-ax, vy=pos[c*3+1]-ay, vz=pos[c*3+2]-az;
    let nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    const l=Math.hypot(nx,ny,nz);
    if (l < 1e-20) return 0;
    nx/=l; ny/=l; nz/=l;
    fn[f*4]=nx; fn[f*4+1]=ny; fn[f*4+2]=nz; fn[f*4+3]=-(nx*ax+ny*ay+nz*az);
    return l/2;
  }
  function addQ(v, a, b, c, d, w) {
    const o = v*10;
    Q[o]+=w*a*a; Q[o+1]+=w*a*b; Q[o+2]+=w*a*c; Q[o+3]+=w*a*d;
    Q[o+4]+=w*b*b; Q[o+5]+=w*b*c; Q[o+6]+=w*b*d;
    Q[o+7]+=w*c*c; Q[o+8]+=w*c*d; Q[o+9]+=w*d*d;
  }
  let area = 0;
  for (let f = 0; f < nf; f++) {
    const a2 = facePlane(f);
    area += a2;
    for (let k = 0; k < 3; k++)
      addQ(F[f*3+k], fn[f*4], fn[f*4+1], fn[f*4+2], fn[f*4+3], a2);
  }

  /* boundary edges (the neck is cut open) get a plane perpendicular to the
     surface so the rim cannot creep inward as the mesh thins */
  const ecount = new Map();
  const ekey = (a, b) => (a < b ? a * nv + b : b * nv + a);
  for (let f = 0; f < nf; f++) for (let k = 0; k < 3; k++) {
    const a = F[f*3+k], b = F[f*3+(k+1)%3], key = ekey(a, b);
    const e = ecount.get(key);
    if (e === undefined) ecount.set(key, [1, f]); else e[0]++;
  }
  let nBoundary = 0;
  for (const [key, [c, f]] of ecount) {
    if (c !== 1) continue;
    nBoundary++;
    const a = Math.floor(key / nv), b = key % nv;
    let ex=pos[b*3]-pos[a*3], ey=pos[b*3+1]-pos[a*3+1], ez=pos[b*3+2]-pos[a*3+2];
    const el = Math.hypot(ex,ey,ez) || 1;
    ex/=el; ey/=el; ez/=el;
    const nx=fn[f*4], ny=fn[f*4+1], nz=fn[f*4+2];
    let cx=ny*ez-nz*ey, cy=nz*ex-nx*ez, cz=nx*ey-ny*ex;
    const cl=Math.hypot(cx,cy,cz)||1; cx/=cl; cy/=cl; cz/=cl;
    const d = -(cx*pos[a*3]+cy*pos[a*3+1]+cz*pos[a*3+2]);
    const w = 1000 * el * el;
    addQ(a, cx, cy, cz, d, w); addQ(b, cx, cy, cz, d, w);
  }
  log(`  ${nBoundary} boundary edges pinned`);

  function qcost(v, x, y, z) {
    const o = v*10;
    return Q[o]*x*x + 2*Q[o+1]*x*y + 2*Q[o+2]*x*z + 2*Q[o+3]*x
         + Q[o+4]*y*y + 2*Q[o+5]*y*z + 2*Q[o+6]*y
         + Q[o+7]*z*z + 2*Q[o+8]*z + Q[o+9];
  }
  /* cost of collapsing (a,b) uses the summed quadric; evaluate it by writing
     the sum into a scratch slot rather than allocating */
  const SQ = new Float64Array(10);
  function sumQ(a, b) { for (let i = 0; i < 10; i++) SQ[i] = Q[a*10+i] + Q[b*10+i]; }
  function sqCost(x, y, z) {
    return SQ[0]*x*x + 2*SQ[1]*x*y + 2*SQ[2]*x*z + 2*SQ[3]*x
         + SQ[4]*y*y + 2*SQ[5]*y*z + 2*SQ[6]*y
         + SQ[7]*z*z + 2*SQ[8]*z + SQ[9];
  }
  const T = new Float64Array(3);
  function best(a, b) {
    sumQ(a, b);
    /* solve the 3x3 for the error-minimising point */
    const m00=SQ[0], m01=SQ[1], m02=SQ[2], m11=SQ[4], m12=SQ[5], m22=SQ[7];
    const c0 = m11*m22 - m12*m12, c1 = m02*m12 - m01*m22, c2 = m01*m12 - m02*m11;
    const det = m00*c0 + m01*c1 + m02*c2;
    if (Math.abs(det) > 1e-14) {
      const b0=-SQ[3], b1=-SQ[6], b2=-SQ[8];
      const x = (b0*c0 + b1*c1 + b2*c2) / det;
      const y = (b0*c1 + b1*(m00*m22-m02*m02) + b2*(m01*m02-m00*m12)) / det;
      const z = (b0*c2 + b1*(m01*m02-m00*m12) + b2*(m00*m11-m01*m01)) / det;
      /* refuse a point that flies off — a near-singular quadric can solve to
         somewhere far outside the neighbourhood and pull a spike out of it */
      const dax=x-pos[a*3], day=y-pos[a*3+1], daz=z-pos[a*3+2];
      const ex=pos[b*3]-pos[a*3], ey=pos[b*3+1]-pos[a*3+1], ez=pos[b*3+2]-pos[a*3+2];
      const el2 = ex*ex+ey*ey+ez*ez;
      if (dax*dax+day*day+daz*daz < el2 * 9 + 1e-12) {
        T[0]=x; T[1]=y; T[2]=z;
        return sqCost(x, y, z);
      }
    }
    let bc = Infinity;
    for (let k = 0; k < 3; k++) {
      const x = k===0 ? pos[a*3]   : k===1 ? pos[b*3]   : (pos[a*3]+pos[b*3])/2;
      const y = k===0 ? pos[a*3+1] : k===1 ? pos[b*3+1] : (pos[a*3+1]+pos[b*3+1])/2;
      const z = k===0 ? pos[a*3+2] : k===1 ? pos[b*3+2] : (pos[a*3+2]+pos[b*3+2])/2;
      const c = sqCost(x, y, z);
      if (c < bc) { bc = c; T[0]=x; T[1]=y; T[2]=z; }
    }
    return bc;
  }

  /* ── heap of candidate collapses, lazily invalidated by vertex stamps ── */
  let cap = 1 << 21, hn = 0;
  let hC = new Float64Array(cap), hA = new Int32Array(cap), hB = new Int32Array(cap),
      hSA = new Int32Array(cap), hSB = new Int32Array(cap);
  function grow() {
    cap *= 2;
    const c=new Float64Array(cap), a=new Int32Array(cap), b=new Int32Array(cap),
          sa=new Int32Array(cap), sb=new Int32Array(cap);
    c.set(hC); a.set(hA); b.set(hB); sa.set(hSA); sb.set(hSB);
    hC=c; hA=a; hB=b; hSA=sa; hSB=sb;
  }
  function push(cost, a, b) {
    if (hn === cap) grow();
    let i = hn++;
    hC[i]=cost; hA[i]=a; hB[i]=b; hSA[i]=stamp[a]; hSB[i]=stamp[b];
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (hC[p] <= hC[i]) break;
      swap(p, i); i = p;
    }
  }
  function swap(i, j) {
    let t=hC[i]; hC[i]=hC[j]; hC[j]=t;
    let u=hA[i]; hA[i]=hA[j]; hA[j]=u;
    u=hB[i]; hB[i]=hB[j]; hB[j]=u;
    u=hSA[i]; hSA[i]=hSA[j]; hSA[j]=u;
    u=hSB[i]; hSB[i]=hSB[j]; hSB[j]=u;
  }
  function pop() {
    if (!hn) return -1;
    const top = 0;
    hn--;
    if (hn > 0) { swap(0, hn); let i = 0;
      for (;;) {
        const l = i*2+1, r = l+1; let s = i;
        if (l < hn && hC[l] < hC[s]) s = l;
        if (r < hn && hC[r] < hC[s]) s = r;
        if (s === i) break;
        swap(s, i); i = s;
      }
    }
    return hn;   // the popped entry now sits at index hn
  }

  for (const key of ecount.keys()) {
    const a = Math.floor(key / nv), b = key % nv;
    push(best(a, b), a, b);
  }
  ecount.clear();
  log(`  ${hn.toLocaleString()} candidate edges`);

  /* neighbours of v, via its faces */
  const nbuf = new Int32Array(256);
  function neighbours(v) {
    let n = 0;
    const fs = vf[v];
    for (let i = 0; i < fs.length; i++) {
      const f = fs[i];
      if (!fAlive[f]) continue;
      for (let k = 0; k < 3; k++) {
        const w = F[f*3+k];
        if (w === v) continue;
        let dup = false;
        for (let j = 0; j < n; j++) if (nbuf[j] === w) { dup = true; break; }
        if (!dup && n < nbuf.length) nbuf[n++] = w;
      }
    }
    return n;
  }

  function compact(v) {
    const fs = vf[v], out = [];
    for (let i = 0; i < fs.length; i++) if (fAlive[fs[i]]) out.push(fs[i]);
    vf[v] = out;
    return out;
  }

  let collapses = 0, rejected = 0;
  const startF = nf;
  while (nf > targetTris && hn > 0) {
    const i = pop();
    const a = hA[i], b = hB[i];
    if (!vAlive[a] || !vAlive[b]) continue;
    if (hSA[i] !== stamp[a] || hSB[i] !== stamp[b]) continue;

    /* recompute — the stored cost is the one that ordered it, T needs filling */
    best(a, b);
    const tx = T[0], ty = T[1], tz = T[2];

    /* link condition: a and b may share exactly as many neighbours as they
       share faces, or the collapse tears the surface into a non-manifold */
    const na = neighbours(a);
    const an = nbuf.slice(0, na);
    const nb = neighbours(b);
    let shared = 0;
    for (let j = 0; j < nb; j++) for (let k = 0; k < na; k++) if (nbuf[j] === an[k]) { shared++; break; }
    let sharedFaces = 0;
    const fa = compact(a);
    for (let j = 0; j < fa.length; j++) {
      const f = fa[j];
      if (F[f*3] === b || F[f*3+1] === b || F[f*3+2] === b) sharedFaces++;
    }
    if (shared !== sharedFaces) { rejected++; continue; }

    /* would any surviving face turn inside out? */
    let flip = false;
    for (const v of [a, b]) {
      const fs = compact(v);
      for (let j = 0; j < fs.length && !flip; j++) {
        const f = fs[j];
        if (F[f*3] === (v===a?b:a) || F[f*3+1] === (v===a?b:a) || F[f*3+2] === (v===a?b:a)) continue;
        const i0=F[f*3], i1=F[f*3+1], i2=F[f*3+2];
        const gx=(v0)=> v0===a||v0===b ? tx : pos[v0*3];
        const gy=(v0)=> v0===a||v0===b ? ty : pos[v0*3+1];
        const gz=(v0)=> v0===a||v0===b ? tz : pos[v0*3+2];
        const ux=gx(i1)-gx(i0), uy=gy(i1)-gy(i0), uz=gz(i1)-gz(i0);
        const vx=gx(i2)-gx(i0), vy=gy(i2)-gy(i0), vz=gz(i2)-gz(i0);
        const nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
        const l = Math.hypot(nx,ny,nz);
        if (l < 1e-18) { flip = true; break; }
        if ((nx*fn[f*4] + ny*fn[f*4+1] + nz*fn[f*4+2]) / l < 0.02) flip = true;
      }
      if (flip) break;
    }
    if (flip) { rejected++; continue; }

    /* do it: a becomes the merged vertex, b dies */
    pos[a*3]=tx; pos[a*3+1]=ty; pos[a*3+2]=tz;
    for (let k = 0; k < 10; k++) Q[a*10+k] += Q[b*10+k];
    const fb = compact(b);
    for (let j = 0; j < fb.length; j++) {
      const f = fb[j];
      if (F[f*3] === a || F[f*3+1] === a || F[f*3+2] === a) { fAlive[f] = 0; nf--; continue; }
      for (let k = 0; k < 3; k++) if (F[f*3+k] === b) F[f*3+k] = a;
      vf[a].push(f);
    }
    vAlive[b] = 0;
    vf[b] = [];
    stamp[a]++; stamp[b]++;
    collapses++;

    const fs = compact(a);
    for (let j = 0; j < fs.length; j++) facePlane(fs[j]);
    const nn = neighbours(a);
    for (let j = 0; j < nn; j++) push(best(a, nbuf[j]), a, nbuf[j]);
  }
  log(`  ${collapses.toLocaleString()} collapses, ${rejected.toLocaleString()} rejected, ` +
      `${startF.toLocaleString()} → ${nf.toLocaleString()} triangles`);

  /* compact to fresh arrays */
  const remap = new Int32Array(nv).fill(-1);
  const outPos = [];
  for (let v = 0; v < nv; v++) {
    if (!vAlive[v]) continue;
    remap[v] = outPos.length / 3;
    outPos.push(pos[v*3], pos[v*3+1], pos[v*3+2]);
  }
  const outInd = [];
  for (let f = 0; f < F.length / 3; f++) {
    if (!fAlive[f]) continue;
    outInd.push(remap[F[f*3]], remap[F[f*3+1]], remap[F[f*3+2]]);
  }
  return { pos: Float64Array.from(outPos), ind: Int32Array.from(outInd) };
}

t0 = Date.now();
const D = decimate(W.pos, W.ind, TARGET);
log(`decimated in ${((Date.now()-t0)/1000).toFixed(1)} s`);

/* ── stitch the kept primitives back on ───────────────────────────────── */
const parts = [D, ...keep.map((g) => ({ pos: g.pos, ind: g.ind }))];
let NV = 0, NI = 0;
for (const p of parts) { NV += p.pos.length / 3; NI += p.ind.length; }
const POS = new Float32Array(NV * 3), IND = new Uint32Array(NI);
{
  let vo = 0, io = 0;
  for (const p of parts) {
    for (let i = 0; i < p.pos.length; i++) POS[vo * 3 + i] = p.pos[i];
    for (let i = 0; i < p.ind.length; i++) IND[io + i] = p.ind[i] + vo;
    vo += p.pos.length / 3; io += p.ind.length;
  }
}
/* normals recomputed on the decimated surface, area weighted */
const NOR = new Float32Array(NV * 3);
for (let f = 0; f < NI; f += 3) {
  const a = IND[f], b = IND[f+1], c = IND[f+2];
  const ux=POS[b*3]-POS[a*3], uy=POS[b*3+1]-POS[a*3+1], uz=POS[b*3+2]-POS[a*3+2];
  const vx=POS[c*3]-POS[a*3], vy=POS[c*3+1]-POS[a*3+1], vz=POS[c*3+2]-POS[a*3+2];
  const nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
  for (const v of [a, b, c]) { NOR[v*3]+=nx; NOR[v*3+1]+=ny; NOR[v*3+2]+=nz; }
}
for (let v = 0; v < NV; v++) {
  const l = Math.hypot(NOR[v*3], NOR[v*3+1], NOR[v*3+2]) || 1;
  NOR[v*3]/=l; NOR[v*3+1]/=l; NOR[v*3+2]/=l;
}
log(`output: ${NV.toLocaleString()} vertices, ${(NI/3).toLocaleString()} triangles`);

/* ── write the GLB ─────────────────────────────────────────────────────── */
function bounds(arr, n) {
  const mn = [Infinity,Infinity,Infinity], mx = [-Infinity,-Infinity,-Infinity];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
    const v = arr[i*3+k];
    if (v < mn[k]) mn[k] = v;
    if (v > mx[k]) mx[k] = v;
  }
  return [mn, mx];
}
const [pmin, pmax] = bounds(POS, NV);
const pad4 = (n) => (n + 3) & ~3;

/* Decimation is the whole point of this file, so the result rarely needs more
   than 65,535 vertices — and at that size a 32-bit index is half zero bytes.
   Narrow it when it fits: worth ~440 KB on the 130 mm head. */
const OUT = NV <= 65536 ? Uint16Array.from(IND) : IND;
const IND_TYPE = OUT.BYTES_PER_ELEMENT === 2 ? 5123 : 5125;

const posBytes = POS.byteLength, norBytes = NOR.byteLength, indBytes = OUT.byteLength;
const binLen = pad4(posBytes) + pad4(norBytes) + pad4(indBytes);
const bin = Buffer.alloc(binLen);
Buffer.from(POS.buffer, POS.byteOffset, posBytes).copy(bin, 0);
Buffer.from(NOR.buffer, NOR.byteOffset, norBytes).copy(bin, pad4(posBytes));
Buffer.from(OUT.buffer, OUT.byteOffset, indBytes).copy(bin, pad4(posBytes) + pad4(norBytes));

const gltf = {
  asset: { version: "2.0", generator: "prepare-scan.mjs (quadric decimation)",
           extras: G.asset.extras },
  scene: 0,
  scenes: [{ nodes: [0] }],
  nodes: [{ mesh: 0, name: "head" }],
  meshes: [{ name: "head", primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2, material: 0 }] }],
  materials: G.materials,
  accessors: [
    { bufferView: 0, componentType: 5126, count: NV, type: "VEC3", min: pmin, max: pmax },
    { bufferView: 1, componentType: 5126, count: NV, type: "VEC3" },
    { bufferView: 2, componentType: IND_TYPE, count: NI, type: "SCALAR" }
  ],
  bufferViews: [
    { buffer: 0, byteOffset: 0, byteLength: posBytes, target: 34962 },
    { buffer: 0, byteOffset: pad4(posBytes), byteLength: norBytes, target: 34962 },
    { buffer: 0, byteOffset: pad4(posBytes) + pad4(norBytes), byteLength: indBytes, target: 34963 }
  ],
  buffers: [{ byteLength: binLen }]
};
let jsonBuf = Buffer.from(JSON.stringify(gltf), "utf8");
if (jsonBuf.length % 4) jsonBuf = Buffer.concat([jsonBuf, Buffer.alloc(4 - jsonBuf.length % 4, 0x20)]);
const out = Buffer.alloc(12 + 8 + jsonBuf.length + 8 + bin.length);
out.write("glTF", 0, "ascii"); out.writeUInt32LE(2, 4);
out.writeUInt32LE(out.length, 8);
out.writeUInt32LE(jsonBuf.length, 12); out.writeUInt32LE(0x4e4f534a, 16);
jsonBuf.copy(out, 20);
out.writeUInt32LE(bin.length, 20 + jsonBuf.length); out.writeUInt32LE(0x004e4942, 24 + jsonBuf.length);
bin.copy(out, 28 + jsonBuf.length);

const dst = path.join(path.dirname(SRC), path.basename(SRC, ".glb") + ".min.glb");
fs.writeFileSync(dst, out);
log(`wrote ${dst} — ${(out.length/1048576).toFixed(2)} MB ` +
    `(was ${(fs.statSync(SRC).size/1048576).toFixed(2)} MB)`);

/* ── the silhouette table the SVGs are drawn from ──────────────────────
   The page draws its busts long before (and on screens that never load) the
   3D model, so the scan's outline has to travel as data inside the HTML: the
   widest reach from the object's own vertical axis at LV heights and BR
   bearings, as a fraction of the object's height. Sampled off the full
   resolution mesh, not the decimated one. */
const LV = 96, BR = 72;
{
  const P = W.pos;                       // welded shell, world space
  const nv = P.length / 3;
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity,z0=Infinity,z1=-Infinity;
  for (let i = 0; i < nv; i++) {
    const x=P[i*3], y=P[i*3+1], z=P[i*3+2];
    if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; if(z<z0)z0=z; if(z>z1)z1=z;
  }
  const H = y1 - y0, cx = (x0+x1)/2, cz = (z0+z1)/2, TAU = Math.PI*2;
  const tab = new Float64Array(LV * BR);
  for (let i = 0; i < nv; i++) {
    const x = (P[i*3]-cx)/H, y = (P[i*3+1]-y1)/H, z = (P[i*3+2]-cz)/H;
    let l = (-y * LV) | 0; if (l < 0) l = 0; else if (l >= LV) l = LV-1;
    let b = ((Math.atan2(z, x) + TAU) % TAU) / TAU * BR | 0; if (b >= BR) b = 0;
    const r = Math.hypot(x, z), k = l*BR + b;
    if (r > tab[k]) tab[k] = r;
  }
  /* bridge bearings the scan had nothing to say about (the crown, mostly) */
  for (let l = 0; l < LV; l++) {
    const row = l*BR;
    let any = false;
    for (let b = 0; b < BR; b++) if (tab[row+b] > 0) { any = true; break; }
    if (!any) { if (l) tab.copyWithin(row, row-BR, row); continue; }
    for (let b = 0; b < BR; b++) {
      if (tab[row+b] > 0) continue;
      let back = 1, fwd = 1;
      while (tab[row + (b-back+BR*2)%BR] === 0) back++;
      while (tab[row + (b+fwd)%BR] === 0) fwd++;
      const a = tab[row + (b-back+BR*2)%BR], c = tab[row + (b+fwd)%BR];
      tab[row+b] = a + (c-a) * (back/(back+fwd));
    }
  }
  let RMAX = 0;
  for (let i = 0; i < tab.length; i++) if (tab[i] > RMAX) RMAX = tab[i];
  const bytes = Buffer.alloc(LV*BR);
  for (let i = 0; i < tab.length; i++) bytes[i] = Math.round(tab[i]/RMAX*255);

  const prof = path.join(import.meta.dirname, "scan-profile.txt");
  fs.writeFileSync(prof,
    `/* ${LV} heights x ${BR} bearings, radius/height, byte of RMAX */\n` +
    `LV=${LV} BR=${BR} RMAX=${RMAX.toFixed(5)}\n` +
    bytes.toString("base64") + "\n");
  log(`wrote ${prof} — ${LV}x${BR}, RMAX ${RMAX.toFixed(5)} of height, ` +
      `${bytes.toString("base64").length} chars`);

  /* front/back reach down the object, so the chin can be located by eye */
  const at = (l, deg) => tab[l*BR + Math.round((deg/360)*BR)%BR];
  log("\n  u     front   back    (bearing 90 = the nose)");
  for (let l = 0; l < LV; l += 3)
    log(`  ${(l/LV).toFixed(3)}  ${at(l,90).toFixed(4)}  ${at(l,270).toFixed(4)}`);
  log(`\n  widest half-width ${Math.max(...Array.from({length:LV},(_,l)=>
    Math.max(at(l,0), at(l,180)))).toFixed(4)} of height`);
}
