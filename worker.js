"use strict";
// No WASM, GPU, shared memory, or threads. A worker owns exactly its role's layers.
const f = Math.fround, D = 384, FF = 1536, S = 512;
let weights = {}, layers = [], role = "", expected = 0, manifest;
const tmp = {n: new Float32Array(D), qkv: new Float32Array(3 * D),
  z: new Float32Array(D), y: new Float32Array(D), ff: new Float32Array(FF),
  scores: new Float32Array(S), logits: new Float32Array(256)};
const send = (m, t = []) => self.postMessage(m, t);

function mat(w, b, x, out) {
  const n = x.length;
  for (let r = 0, off = 0; r < out.length; r++, off += n) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let j = 0; j < n; j += 4) {
      s0 = f(s0 + f(w[off+j] * x[j]));
      s1 = f(s1 + f(w[off+j+1] * x[j+1]));
      s2 = f(s2 + f(w[off+j+2] * x[j+2]));
      s3 = f(s3 + f(w[off+j+3] * x[j+3]));
    }
    out[r] = f(f(f(s0 + s1) + f(s2 + s3)) + (b ? b[r] : 0));
  }
}
function norm(x, prefix, out) {
  const w = weights[prefix+".weight"], b = weights[prefix+".bias"];
  let mean = 0, variance = 0;
  for (let i = 0; i < D; i++) mean = f(mean + x[i]);
  mean = f(mean / D);
  for (let i = 0; i < D; i++) { const d = f(x[i] - mean); variance = f(variance + f(d*d)); }
  const inv = f(1 / Math.sqrt(f(f(variance / D) + f(1e-5))));
  for (let i = 0; i < D; i++) out[i] = f(f(f(x[i]-mean)*inv)*w[i]+b[i]);
}
function erf(x) {
  // Positive convergent series: erf(x)=2*x*exp(-x*x)/sqrt(pi) *
  // sum_n (2*x*x)^n/(1*3*...*(2*n+1)). No cancellation; double precision.
  const a = Math.abs(x);
  if (a >= 6) return Math.sign(x); // discarded tail < 2.2e-17
  let term = 1, sum = 1;
  for (let n = 1; n < 300; n++) {
    term *= 2*a*a/(2*n+1); sum += term;
    if (term < sum*1e-17) break;
  }
  return Math.sign(x)*1.1283791670955126*a*Math.exp(-a*a)*sum;
}
function block(x, layer, pos) {
  const p = "blocks."+layer.i, qkv = tmp.qkv, z = tmp.z, sc = tmp.scores;
  norm(x, p+".ln1", tmp.n);
  mat(weights[p+".qkv.weight"], weights[p+".qkv.bias"], tmp.n, qkv);
  layer.k.set(qkv.subarray(D, 2*D), pos*D);
  layer.v.set(qkv.subarray(2*D), pos*D);
  for (let h = 0; h < 6; h++) {
    const base = h*64;
    let max = -Infinity;
    for (let t = 0; t <= pos; t++) {
      let dot = 0; const off = t*D+base;
      for (let j = 0; j < 64; j++) dot = f(dot + f(qkv[base+j]*layer.k[off+j]));
      sc[t] = f(dot*0.125); max = Math.max(max, sc[t]);
    }
    let sum = 0;
    for (let t = 0; t <= pos; t++) { sc[t] = f(Math.exp(f(sc[t]-max))); sum = f(sum+sc[t]); }
    for (let t = 0; t <= pos; t++) sc[t] = f(sc[t]/sum);
    for (let j = 0; j < 64; j++) {
      let val = 0;
      for (let t = 0; t <= pos; t++) val = f(val+f(sc[t]*layer.v[t*D+base+j]));
      z[base+j] = val;
    }
  }
  mat(weights[p+".proj.weight"], weights[p+".proj.bias"], z, tmp.y);
  for (let i = 0; i < D; i++) x[i] = f(x[i]+tmp.y[i]);
  norm(x, p+".ln2", tmp.n);
  mat(weights[p+".fc1.weight"], weights[p+".fc1.bias"], tmp.n, tmp.ff);
  for (let i = 0; i < FF; i++) { const a = tmp.ff[i]; tmp.ff[i] = f(0.5*a*(1+erf(a*Math.SQRT1_2))); }
  mat(weights[p+".fc2.weight"], weights[p+".fc2.bias"], tmp.ff, tmp.y);
  for (let i = 0; i < D; i++) x[i] = f(x[i]+tmp.y[i]);
  return x;
}
function embed(byte, pos) {
  const x = new Float32Array(D), t = weights["tok_emb.weight"], p = weights["pos_emb.weight"];
  for (let i = 0; i < D; i++) x[i] = f(t[byte*D+i]+p[pos*D+i]);
  return x;
}
function choose(x, temperature) {
  norm(x, "ln_f", tmp.n);
  mat(weights["lm_head.weight"], null, tmp.n, tmp.logits);
  let best = 0;
  for (let i = 1; i < 256; i++) if (tmp.logits[i] > tmp.logits[best]) best = i;
  if (!temperature) return best;
  let total = 0; const probs = new Float64Array(256), max = tmp.logits[best];
  for (let i = 0; i < 256; i++) total += probs[i] = Math.exp((tmp.logits[i]-max)/temperature);
  const random = new Uint32Array(1); crypto.getRandomValues(random);
  let r = (random[0]/4294967296)*total;
  for (let i = 0; i < 256; i++) { r -= probs[i]; if (r < 0) return i; }
  return 255;
}
async function cached(name) {
  const url = new URL(name, self.location.href).href;
  const cache = await caches.open("split-gpt-data-v1");
  let response = await cache.match(url);
  if (!response) {
    response = await fetch(url, {cache:"no-cache"});
    if (!response.ok) throw Error(name+": HTTP "+response.status);
    await cache.put(url, response.clone());
  }
  return response;
}
const sha = async b => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", b)))
  .map(x=>x.toString(16).padStart(2,"0")).join("");
async function load(part) {
  if (!["a","b","test"].includes(part)) throw Error("Invalid role");
  role = part;
  manifest = await (await cached("weights_manifest.json")).json();
  if (manifest.params !== 11040768 || manifest.dtype !== "float32-le") throw Error("Wrong manifest");
  weights = {};
  const names = part === "test" ? ["part_a.bin","part_b.bin"] : ["part_"+part+".bin"];
  for (const file of names) {
    const buffer = await (await cached(file)).arrayBuffer();
    if (buffer.byteLength !== manifest.files[file].bytes || await sha(buffer) !== manifest.files[file].sha256)
      throw Error("Weight integrity failed. Clear site data and reload: "+file);
    const little = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
    for (const t of manifest.tensors.filter(t=>t.part===file)) {
      let a;
      if (little) a = new Float32Array(buffer, t.part_offset, t.nbytes/4);
      else { a = new Float32Array(t.nbytes/4); const v = new DataView(buffer);
        for (let i=0;i<a.length;i++) a[i]=v.getFloat32(t.part_offset+i*4,true); }
      weights[t.name] = a;
    }
  }
  const ids = part === "a" ? [0,1,2] : part === "b" ? [3,4,5] : [0,1,2,3,4,5];
  layers = ids.map(i=>({i,k:new Float32Array(S*D),v:new Float32Array(S*D)}));
  expected = 0;
  send({type:"ready", model:manifest.model_id, layers:ids});
}
function step(m) {
  if (m.pos !== expected || m.pos >= S) throw Error("Out-of-order/overflow position");
  const start = performance.now();
  let x = role === "a" ? embed(m.byte, m.pos) : new Float32Array(m.buffer);
  if (x.length !== D) throw Error("Invalid activation");
  for (const l of layers) block(x,l,m.pos);
  expected++;
  if (role === "a") send({type:"hidden",run:m.run,pos:m.pos,ms:performance.now()-start,buffer:x.buffer},[x.buffer]);
  else send({type:"token",run:m.run,pos:m.pos,byte:choose(x,m.temperature),ms:performance.now()-start});
}
async function selfTest() {
  if (role !== "test") throw Error("Full model is restricted to ?selftest=1");
  const ref = await (await cached("reference.json")).json();
  if (ref.model_id !== manifest.model_id) throw Error("Reference model mismatch");
  expected = 0;
  let first, token, pos = 0;
  const out = [], start = performance.now();
  const run = b => {
    let x = embed(b,pos);
    for (const l of layers) { block(x,l,pos); if(pos===0 && l.i===0) first=Array.from(x); }
    pos++; return choose(x,0);
  };
  for (const b of ref.prompt) token = run(b);
  for (let i=0;i<ref.bytes.length;i++) { out.push(token); if(i+1<ref.bytes.length) token=run(token); }
  let error=0, sum=0;
  for(let i=0;i<D;i++) { const d=first[i]-ref.first_layer[i]; error=Math.max(error,Math.abs(d)); sum+=d*d; }
  send({type:"test", exact:out.every((b,i)=>b===ref.bytes[i]), maxAbs:error,
    rms:Math.sqrt(sum/D), bytes:out, first_layer:first, ms:performance.now()-start,
    reference:ref.bytes, minMargin:ref.min_argmax_margin});
}
self.onmessage = async e => {
  try {
    const m = e.data;
    if (m.type === "load") await load(m.role);
    else if (m.type === "reset") expected = 0; // Subsequent positions overwrite all visible KV entries.
    else if (m.type === "step") { if(!["a","b"].includes(role)) throw Error("Invalid inference role"); step(m); }
    else if (m.type === "test") await selfTest();
  } catch (e) { send({type:"error",error:e.message}); }
};
