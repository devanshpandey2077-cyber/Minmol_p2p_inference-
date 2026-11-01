// Optional desktop verification; Node 20+ standard library only.
import fs from "node:fs/promises";
import vm from "node:vm";
import assert from "node:assert/strict";
import {webcrypto} from "node:crypto";
import {performance} from "node:perf_hooks";
const source=await fs.readFile(new URL("worker.js",import.meta.url),"utf8");
const reference=JSON.parse(await fs.readFile(new URL("reference.json",import.meta.url),"utf8"));
async function engine(role){
  const messages=[],reads=[];
  const context=vm.createContext({Float32Array,Float64Array,Uint8Array,Uint32Array,DataView,
    TextEncoder,TextDecoder,URL,crypto:webcrypto,performance,
    caches:{open:async()=>({match:async()=>null,put:async()=>{}})},
    fetch:async url=>{const file=new URL(url);reads.push(file.pathname);return new Response(await fs.readFile(file));},
    self:{location:{href:new URL("worker.js",import.meta.url).href},postMessage:m=>messages.push(m)}});
  vm.runInContext(source,context);
  const send=async m=>{messages.length=0;await context.self.onmessage({data:m});
    const result=messages.at(-1);if(result?.type==="error")throw Error(result.error);return result;};
  await send({type:"load",role});return {send,reads};
}
const full=await engine("test"),test=await full.send({type:"test"});
assert.equal(test.exact,true,"JS/PyTorch greedy bytes differ");
assert.ok(test.maxAbs<1e-4,"First block activation tolerance exceeded");
console.log("PyTorch parity:",JSON.stringify({exact:test.exact,maxAbs:test.maxAbs,rms:test.rms,minMargin:test.minMargin}));
const a=await engine("a"),b=await engine("b");
assert.ok(!a.reads.some(x=>x.endsWith("part_b.bin")||x.endsWith("weights.bin")));
assert.ok(!b.reads.some(x=>x.endsWith("part_a.bin")||x.endsWith("weights.bin")));
async function split(prompt,n){
  await a.send({type:"reset"});await b.send({type:"reset"});
  let pos=0,token;const out=[];
  async function step(byte){
    const hidden=await a.send({type:"step",run:1,pos,byte});
    assert.equal(hidden.buffer.byteLength,1536);
    const reply=await b.send({type:"step",run:1,pos,buffer:hidden.buffer,temperature:0});pos++;return reply.byte;
  }
  for(const byte of prompt)token=await step(byte);
  for(let i=0;i<n;i++){out.push(token);if(i+1<n)token=await step(token);}
  return out;
}
assert.deepEqual(await split(reference.prompt,reference.bytes.length),reference.bytes);
assert.deepEqual(await split(reference.benchmark.prompt,128),reference.benchmark.bytes);
console.log("Split parity: reference + 128-byte benchmark; half-only downloads passed");
const qc=vm.createContext({TextEncoder,TextDecoder,Uint8Array,Uint32Array,window:{}});
vm.runInContext(await fs.readFile(new URL("qr.js",import.meta.url),"utf8"),qc);
const qr=qc.window.QR;
const text="HG1:01234567:1:12:HG1.01234567.ThisIsAQRFrameWith0123456789_abcdefghijklmnopqrstuvwxyz";
const grid=qr.encode(text);assert.equal(qr.decodeGrid(grid),text);
// Damage 13 separate codewords in the first data stripe (one bit each).
const damaged=grid.map(r=>Uint8Array.from(r));
let codeIndex=0,up=true,flips=0;
// First stripe x=36,35 has 28 non-function rows y=36..9, hence 56 bits.
// Flip once in each of the first 7 bytes, then use the next stripe's lower rows.
for(let y=36;y>=9;y--)for(const x of [36,35]){if(codeIndex%8===0&&flips<7){damaged[y][x]^=1;flips++;}codeIndex++;}
// The next stripe starts at bottom with upward=false: scan y=9..36.
for(let y=9;y<=36;y++)for(const x of [34,33]){if(codeIndex%8===0&&flips<13){damaged[y][x]^=1;flips++;}codeIndex++;}
assert.equal(qr.decodeGrid(damaged),text);
function raster(map,width=450,height=450){
  const data=new Uint8ClampedArray(width*height*4);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const [u,v]=map(x,y),gx=Math.floor(u),gy=Math.floor(v);
    const value=gx>=0&&gy>=0&&gx<37&&gy<37&&grid[gy][gx]?0:255,j=(y*width+x)*4;
    data[j]=data[j+1]=data[j+2]=value;data[j+3]=255;
  }return {width,height,data};
}
assert.equal(qr.scan(raster((x,y)=>[x/10-4,y/10-4])),text);
// Affine shear/rotation tests exercise finder detection away from axis alignment.
assert.equal(qr.scan(raster((x,y)=>[(x-52-0.12*(y-35))/8,(y-35)/9],440,440)),text);
assert.equal(qr.scan(raster((x,y)=>[(y-40)/9,(410-x)/9],450,450)),text);
console.log("QR: grid, 13-codeword correction, raster, shear, rotation passed");
// Exercise signaling codec without a DOM or browser.
const app=await fs.readFile(new URL("app.js",import.meta.url),"utf8");
const section=app.slice(app.indexOf("function compress("),app.indexOf("function showFrame("));
const ac=vm.createContext({Uint8Array,TextEncoder,TextDecoder,btoa,atob,
  enc:new TextEncoder(),dec:new TextDecoder(),fnv:bytes=>{let h=2166136261;for(const b of bytes)h=Math.imul(h^b,16777619)>>>0;return h.toString(16).padStart(8,"0");}});
vm.runInContext(section,ac);
for(let n=1;n<1000;n+=17){const data=Uint8Array.from({length:n},(_,i)=>(i*71+i*i)%256);
  assert.deepEqual(Array.from(ac.decompress(ac.compress(data))),Array.from(data));}
const description={type:"offer",sdp:"v=0\r\na=candidate:1 1 udp 2113937151 abc.local 41234 typ host\r\n".repeat(8)};
assert.deepEqual(JSON.parse(JSON.stringify(ac.decodeSDP(ac.encodeSDP(description)))),description);
console.log("Signaling: LZSS and SDP round trips passed");
// Wire two independent app controllers and real inference workers through an
// asynchronous in-memory reliable channel. This tests protocol/control flow,
// not the device's actual ICE/camera implementation.
function controller(){
  const elements=new Map();
  const get=id=>{if(!elements.has(id))elements.set(id,{value:id==="prompt"?"Hello":id==="maxBytes"?"32":"0",hidden:false,
    classList:{toggle(){}},textContent:""});return elements.get(id);};
  class LocalWorker {
    constructor(){this.chain=Promise.resolve();this.role=null;}
    postMessage(m){
      if(m.type==="load")this.chain=engine(m.role).then(e=>{this.role=m.role;this.engine=e;
        this.onmessage({data:{type:"ready",model:reference.model_id,layers:m.role==="a"?[0,1,2]:[3,4,5]}});return e;});
      else this.chain=this.chain.then(async e=>{const result=await e.send(m);if(result)this.onmessage({data:result});return e;});
      this.chain.catch(error=>this.onerror?.({message:error.message}));
    }
  }
  const context=vm.createContext({document:{getElementById:get},TextEncoder,TextDecoder,Uint8Array,DataView,ArrayBuffer,
    Worker:LocalWorker,navigator:{},performance,setTimeout,clearTimeout,setInterval,clearInterval,console,
    btoa,atob,URL,URLSearchParams,location:{search:""}});
  vm.runInContext(app.slice(0,app.indexOf("const action=")),context);return {context,get};
}
const ca=controller(),cb=controller();
vm.runInContext('selectRole("a")',ca.context);vm.runInContext('selectRole("b")',cb.context);
ca.context.link=buffer=>setImmediate(()=>{cb.context.packet=buffer;vm.runInContext("receive(packet)",cb.context);});
cb.context.link=buffer=>setImmediate(()=>{ca.context.packet=buffer;vm.runInContext("receive(packet)",ca.context);});
for(const c of [ca,cb])vm.runInContext('dc={readyState:"open",send:link}',c.context);
await new Promise((resolve,reject)=>{
  const timeout=setTimeout(()=>{clearInterval(timer);reject(Error("Controller test timeout"));},60000);
  const timer=setInterval(()=>{
    try{
      for(const c of [ca,cb])if(c.get("status").classList.error)throw Error(c.get("status").textContent);
      if(vm.runInContext("peerReady",ca.context)&&vm.runInContext("peerReady",cb.context)){
        clearInterval(timer);clearTimeout(timeout);resolve();}
    }catch(e){clearInterval(timer);clearTimeout(timeout);reject(e);}
  },10);
});
vm.runInContext("start(true)",ca.context);
await new Promise((resolve,reject)=>{
  const timeout=setTimeout(()=>{clearInterval(timer);reject(Error("Generation protocol timeout"));},60000);
  const timer=setInterval(()=>{
    if(vm.runInContext("run===null",ca.context)&&vm.runInContext("run===null",cb.context)){
      clearInterval(timer);clearTimeout(timeout);resolve();}
  },10);
});
for(const c of [ca,cb])assert.deepEqual(JSON.parse(c.get("byteValues").textContent),reference.benchmark.bytes);
console.log("Controllers: two-worker binary protocol and live output checksums passed");
