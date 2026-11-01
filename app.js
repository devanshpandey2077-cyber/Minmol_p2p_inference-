"use strict";
const $=id=>document.getElementById(id),enc=new TextEncoder(),dec=new TextDecoder();
const T={HELLO:1,START:2,HIDDEN:3,TOKEN:4,DONE:5,ERROR:6,STATS:7};
let role,pc,dc,worker,ready=false,peerReady=false,peerHello=null,model,run=null,serial=0,pending=null,deadline;
let frames=[],frameIndex=0,frameTimer,stream,scanWorker,scanning=false,scanBusy=false,collections=new Map();
const fnv=bytes=>{let h=2166136261;for(const b of bytes)h=Math.imul(h^b,16777619)>>>0;return h.toString(16).padStart(8,"0");};
function status(s,error=false){$("status").textContent=s;$("status").classList.toggle("error",error);}
function fail(e){
  clearTimeout(deadline);status(e.message||String(e),true);
  if(run){run=null;$("runStatus").textContent="Run interrupted; start a new run to reset both KV caches.";}
  if(dc?.readyState==="open")try{control(T.ERROR,{error:e.message||String(e)});}catch(e){}
  buttons();
}
function buttons(){
  const can=role==="a"&&ready&&peerReady&&dc?.readyState==="open"&&!run;
  $("generate").disabled=$("benchmark").disabled=!can;$("stop").disabled=!run||role!=="a";
  for(const id of ["prompt","temperature","maxBytes"])$(id).disabled=role!=="a"||!!run;
  $("makeOffer").disabled=!!pc;$("apply").disabled=!!run||dc?.readyState==="open";
}
// LZSS: groups of eight tokens, control bit 1 = literal; otherwise 12-bit
// backward distance-1 and 4-bit length-3. Every browser uses the same codec.
function compress(bytes){
  const out=[];
  for(let i=0;i<bytes.length;){
    const at=out.length;out.push(0);let flags=0;
    for(let bit=0;bit<8&&i<bytes.length;bit++){
      let length=0,distance=0;
      for(let j=i-1;j>=Math.max(0,i-4096);j--){
        if(bytes[j]!==bytes[i])continue;let n=1;
        while(n<18&&i+n<bytes.length&&bytes[j+n]===bytes[i+n])n++;
        if(n>length){length=n;distance=i-j;if(n===18)break;}
      }
      if(length>=3){const v=((distance-1)<<4)|(length-3);out.push(v>>>8,v&255);i+=length;}
      else{flags|=1<<bit;out.push(bytes[i++]);}
    }out[at]=flags;
  }return Uint8Array.from(out);
}
function decompress(bytes){
  const out=[];let i=0;
  while(i<bytes.length){const flags=bytes[i++];
    for(let bit=0;bit<8&&i<bytes.length;bit++){
      if(flags&(1<<bit))out.push(bytes[i++]);
      else{if(i+1>=bytes.length)throw Error("Truncated pairing text");const v=(bytes[i++]<<8)|bytes[i++],d=(v>>>4)+1,n=(v&15)+3;
        if(d>out.length)throw Error("Invalid pairing text");for(let j=0;j<n;j++)out.push(out[out.length-d]);}
      if(out.length>65536)throw Error("Pairing text too large");
    }
  }return Uint8Array.from(out);
}
function encodeSDP(description){
  const raw=enc.encode(JSON.stringify({type:description.type,sdp:description.sdp})),zip=compress(raw);
  return "HG1."+fnv(raw)+"."+btoa(String.fromCharCode(...zip)).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");
}
function decodeSDP(text){
  const m=text.trim().match(/^HG1\.([0-9a-f]{8})\.([A-Za-z0-9_-]+)$/);if(!m||m[2].length>90000)throw Error("Invalid pairing text");
  const zip=Uint8Array.from(atob(m[2].replace(/-/g,"+").replace(/_/g,"/")),c=>c.charCodeAt(0)),raw=decompress(zip);
  if(fnv(raw)!==m[1])throw Error("Pairing checksum mismatch");
  const d=JSON.parse(dec.decode(raw));if(!["offer","answer"].includes(d.type)||typeof d.sdp!=="string")throw Error("Invalid SDP");return d;
}
function showFrame(){
  if(!frames.length)return;frameIndex=(frameIndex+frames.length)%frames.length;
  QR.draw($("qr"),frames[frameIndex]);$("frameLabel").textContent=`Frame ${frameIndex+1}/${frames.length} · auto-cycles every 1.2 s`;
}
function showPairing(description){
  const text=encodeSDP(description);$("local").value=text;
  const id=fnv(enc.encode(text)),count=Math.ceil(text.length/78);if(count>999)throw Error("Too many QR frames");
  frames=Array.from({length:count},(_,i)=>`HG1:${id}:${i+1}:${count}:${text.slice(i*78,(i+1)*78)}`);
  frameIndex=0;$("qrBox").hidden=false;showFrame();clearInterval(frameTimer);
  frameTimer=setInterval(()=>{frameIndex++;showFrame();},1200);
}
function control(type,obj){
  const bytes=enc.encode(JSON.stringify(obj)),buffer=new ArrayBuffer(4+bytes.length),v=new Uint8Array(buffer);v[0]=type;v.set(bytes,4);dc.send(buffer);
}
function hello(){if(ready&&dc?.readyState==="open")control(T.HELLO,{model,role});}
function checkPeer(){
  if(!ready||!peerHello)return;
  if(peerHello.model!==model||peerHello.role===role)throw Error("Peer role/weight mismatch");
  peerReady=true;status("Connected · both halves ready · no relay");buttons();
}
function makePeer(){
  if(pc)throw Error("Reload the page to start a new pairing");
  pc=new RTCPeerConnection({iceServers:[],iceTransportPolicy:"all",iceCandidatePoolSize:0});
  pc.ondatachannel=e=>attach(e.channel);
  pc.onconnectionstatechange=()=>{
    status("WebRTC "+pc.connectionState+" · "+(ready?"weights ready":"loading weights"));
    if(["failed","closed"].includes(pc.connectionState))fail(Error("Connection lost. Reload both phones to re-pair."));
  };return pc;
}
async function gather(){
  if(pc.iceGatheringState!=="complete")await new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{cleanup();reject(Error("ICE gathering timed out. Retry on a normal Wi-Fi/hotspot."));},30000);
    const change=()=>{if(pc.iceGatheringState==="complete"){cleanup();resolve();}};
    const cleanup=()=>{clearTimeout(timeout);pc.removeEventListener("icegatheringstatechange",change);};
    pc.addEventListener("icegatheringstatechange",change);change();
  });
  if(!/^a=candidate:/m.test(pc.localDescription.sdp))throw Error("No host ICE candidate. Check browser LAN permissions or Wi-Fi.");
  showPairing(pc.localDescription);
}
function attach(channel){
  if(dc)throw Error("Unexpected second channel");dc=channel;dc.binaryType="arraybuffer";
  if(!dc.ordered||dc.maxRetransmits!==null||dc.maxPacketLifeTime!==null)throw Error("Reliable ordered channel required");
  dc.onopen=()=>{status("Connected directly · waiting for both weight halves");hello();buttons();inspect().catch(fail);};
  dc.onclose=()=>{peerReady=false;fail(Error("DataChannel closed. Reload both phones."));};
  dc.onerror=()=>fail(Error("DataChannel error"));
  dc.onmessage=e=>{try{receive(e.data);}catch(e){fail(e);}};
}
async function inspect(){
  if(!pc)return;const stats=await pc.getStats();let pair;
  stats.forEach(s=>{if(s.type==="transport"&&s.selectedCandidatePairId)pair=stats.get(s.selectedCandidatePairId);});
  if(!pair)stats.forEach(s=>{if(s.type==="candidate-pair"&&s.state==="succeeded"&&s.nominated)pair=s;});
  if(!pair){$("network").textContent="Waiting for selected ICE pair";return;}
  const local=stats.get(pair.localCandidateId),remote=stats.get(pair.remoteCandidateId);
  if(local?.candidateType==="relay"||remote?.candidateType==="relay")throw Error("Relay candidate prohibited");
  $("network").textContent=JSON.stringify({iceServers:pc.getConfiguration().iceServers,
    localType:local?.candidateType,remoteType:remote?.candidateType,
    protocol:local?.protocol,networkRTTms:pair.currentRoundTripTime===undefined?null:pair.currentRoundTripTime*1000,
    bytesSent:pair.bytesSent,bytesReceived:pair.bytesReceived},null,2);
}
function render(){
  if(!run)return;$("output").textContent=dec.decode(Uint8Array.from([...run.prompt,...run.out]));
  $("checksum").textContent=fnv(run.out);$("byteValues").textContent=JSON.stringify(run.out);
}
function displayStats(s){
  $("rate").textContent=s.rate.toFixed(2);$("aMs").textContent=s.a.toFixed(1);$("bMs").textContent=s.b.toFixed(1);$("rtt").textContent=s.rtt.toFixed(1);
}
function stats(){
  const n=run.steps||1,seconds=(performance.now()-run.decodeStart)/1000;
  return {rate:run.out.length/Math.max(seconds,0.001),a:run.a/n,b:run.b/n,rtt:run.rtt/n};
}
function validateStart(s){
  if(!Number.isInteger(s.id)||!Array.isArray(s.prompt)||s.prompt.length<1||!s.prompt.every(x=>Number.isInteger(x)&&x>=0&&x<256)||
    !Number.isInteger(s.n)||s.n<1||s.prompt.length+s.n-1>512||!Number.isFinite(s.temp)||s.temp<0||s.temp>2)throw Error("Invalid generation request");
}
function begin(s){
  validateStart(s);run={...s,out:[],position:0,a:0,b:0,rtt:0,steps:0,started:performance.now(),decodeStart:0,stopping:false};
  worker.postMessage({type:"reset"});render();$("runStatus").textContent="Prefilling "+s.prompt.length+" bytes…";buttons();
}
function dispatch(byte){
  pending={pos:run.position,emit:run.position>=run.prompt.length-1};
  if(pending.emit&&run.out.length===0)run.decodeStart=performance.now();
  worker.postMessage({type:"step",byte,pos:run.position,run:run.id});
  clearTimeout(deadline);deadline=setTimeout(()=>fail(Error("Peer response timed out. Keep both phones foregrounded.")),60000);
}
function finish(summary){
  clearTimeout(deadline);if(!run)return;
  if(summary.hash!==fnv(run.out)||summary.count!==run.out.length)throw Error("Peer output checksum mismatch");
  displayStats(summary.stats);
  $("runStatus").textContent=`${run.bench?"Benchmark":"Generation"} complete · ${summary.count} bytes · ${summary.seconds.toFixed(2)} s including prefill · ${summary.stats.rate.toFixed(2)} bytes/s decoding`;
  run=null;pending=null;buttons();inspect().catch(fail);
}
function receive(buffer){
  if(!(buffer instanceof ArrayBuffer)||buffer.byteLength<4)throw Error("Invalid protocol packet");
  const v=new DataView(buffer),type=v.getUint8(0);
  if(type===T.HIDDEN){
    if(role!=="b"||!run||buffer.byteLength!==1548||v.getUint32(4,true)!==run.id||v.getUint16(2,true)!==run.position)throw Error("Unexpected hidden state");
    const emit=v.getUint8(1)===1;if(emit!==(run.position>=run.prompt.length-1))throw Error("Invalid prefill flag");
    pending={emit,a:v.getFloat32(8,true)};
    if(emit&&run.out.length===0)run.decodeStart=performance.now();
    const payload=buffer.slice(12);worker.postMessage({type:"step",run:run.id,pos:run.position,buffer:payload,temperature:emit?run.temp:0},[payload]);return;
  }
  if(type===T.TOKEN){
    if(role!=="a"||!run||buffer.byteLength!==12||v.getUint32(4,true)!==run.id||v.getUint16(2,true)!==run.position)throw Error("Unexpected token");
    clearTimeout(deadline);const byte=v.getUint8(1),b=v.getFloat32(8,true),rtt=performance.now()-pending.sent;
    if(pending.emit){
      run.out.push(byte);run.a+=pending.a;run.b+=b;run.rtt+=rtt;run.steps++;render();
      const s=stats();displayStats(s);control(T.STATS,s);$("runStatus").textContent=`Generating ${run.out.length}/${run.n} bytes…`;
    }
    if(run.stopping||run.out.length===run.n){
      const summary={hash:fnv(run.out),count:run.out.length,stats:run.out.length?stats():{rate:0,a:0,b:0,rtt:0},seconds:(performance.now()-run.started)/1000};
      control(T.DONE,summary);finish(summary);return;
    }
    run.position++;dispatch(run.position<run.prompt.length?run.prompt[run.position]:byte);return;
  }
  const obj=JSON.parse(dec.decode(new Uint8Array(buffer,4)));
  if(type===T.HELLO){
    peerHello=obj;checkPeer();
  }else if(type===T.START){if(role!=="b"||!ready||!peerReady||run)throw Error("Cannot start peer run");begin(obj);
  }else if(type===T.STATS){if(role==="b"&&run)displayStats(obj);
  }else if(type===T.DONE){if(role!=="b"||!run)throw Error("Unexpected completion");finish(obj);
  }else if(type===T.ERROR){clearTimeout(deadline);run=null;status("Peer: "+obj.error,true);buttons();
  }else throw Error("Unknown packet type");
}
function workerMessage(m){
  if(m.type==="error")throw Error(m.error);
  if(m.type==="ready"){
    ready=true;model=m.model;$("weightStatus").textContent="Weights cached and SHA-256 verified · layers "+m.layers.join(", ");
    hello();checkPeer();buttons();return;
  }
  if(!run||m.run!==run.id)return;
  if(m.type==="hidden"){
    pending.a=m.ms;pending.sent=performance.now();const packet=new ArrayBuffer(1548),v=new DataView(packet);
    v.setUint8(0,T.HIDDEN);v.setUint8(1,pending.emit?1:0);v.setUint16(2,m.pos,true);v.setUint32(4,run.id,true);v.setFloat32(8,m.ms,true);
    new Uint8Array(packet,12).set(new Uint8Array(m.buffer));dc.send(packet);
  }else if(m.type==="token"){
    const packet=new ArrayBuffer(12),v=new DataView(packet);v.setUint8(0,T.TOKEN);v.setUint8(1,m.byte);v.setUint16(2,m.pos,true);v.setUint32(4,run.id,true);v.setFloat32(8,m.ms,true);dc.send(packet);
    if(pending.emit){run.out.push(m.byte);render();$("runStatus").textContent=`Generating ${run.out.length}/${run.n} bytes…`;}
    run.position++;
  }
}
function selectRole(r){
  if(role)return;role=r;$("roles").hidden=true;$("pair").hidden=$("generation").hidden=false;
  $("makeOffer").hidden=r!=="a";$("roleTitle").textContent=r==="a"?"Phone A · host":"Phone B · join";
  status("Loading only part_"+r+".bin…");worker=new Worker("worker.js");
  worker.onmessage=e=>{try{workerMessage(e.data);}catch(e){fail(e);}};worker.onerror=e=>fail(Error(e.message));
  worker.postMessage({type:"load",role:r});buttons();
  if(navigator.storage?.persist)navigator.storage.persist().catch(()=>{});
  if(navigator.wakeLock)navigator.wakeLock.request("screen").catch(()=>{});
}
async function applyText(text){
  const d=decodeSDP(text);
  if(role==="b"){
    if(d.type!=="offer")throw Error("Join requires an offer");makePeer();await pc.setRemoteDescription(d);
    await pc.setLocalDescription(await pc.createAnswer());await gather();status("Answer ready. Scan/copy it back to Phone A.");
  }else{
    if(!pc||d.type!=="answer")throw Error("Create an offer first, then import the answer");
    await pc.setRemoteDescription(d);status("Answer imported. Connecting directly…");
  }
  stopCamera();
}
function acceptFrame(text){
  const m=text.match(/^HG1:([0-9a-f]{8}):(\d+):(\d+):(.+)$/);if(!m)return;
  const index=+m[2],count=+m[3];if(count<1||count>999||index<1||index>count)return;
  let set=collections.get(m[1]);if(!set){set={count,parts:new Map()};collections.set(m[1],set);}
  if(set.count!==count)return;set.parts.set(index,m[4]);$("scanStatus").textContent=`Scanned ${set.parts.size}/${count} frames. Hold steady while the other phone cycles.`;
  if(set.parts.size===count){const full=Array.from({length:count},(_,i)=>set.parts.get(i+1)).join("");
    if(fnv(enc.encode(full))!==m[1])throw Error("QR frame checksum mismatch");$("remote").value=full;stopCamera();applyText(full).catch(fail);}
}
async function startCamera(){
  if(!isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error("Camera requires HTTPS (localhost also works on desktop)");
  stopCamera();collections.clear();
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"},width:{ideal:1280}},audio:false});
  $("camera").srcObject=stream;$("camera").hidden=false;$("stopCamera").hidden=false;await $("camera").play();
  scanning=true;scanWorker=new Worker("qr.js");
  scanWorker.onmessage=e=>{scanBusy=false;if(scanning&&e.data.text)try{acceptFrame(e.data.text);}catch(e){fail(e);}};
  scanWorker.onerror=()=>{scanBusy=false;$("scanStatus").textContent="Scanner failed. Use pairing text.";stopCamera();};
  const canvas=document.createElement("canvas"),ctx=canvas.getContext("2d",{willReadFrequently:true});
  const tick=()=>{
    if(!scanning)return;
    if(!scanBusy&&$("camera").readyState>=2){
      const video=$("camera"),scale=Math.min(1,640/video.videoWidth);canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
      ctx.drawImage(video,0,0,canvas.width,canvas.height);const image=ctx.getImageData(0,0,canvas.width,canvas.height);scanBusy=true;
      scanWorker.postMessage({width:image.width,height:image.height,data:image.data},[image.data.buffer]);
    }setTimeout(tick,180);
  };tick();
}
function stopCamera(){
  scanning=false;scanBusy=false;stream?.getTracks().forEach(t=>t.stop());stream=null;scanWorker?.terminate();scanWorker=null;
  $("camera").srcObject=null;$("camera").hidden=true;$("stopCamera").hidden=true;
}
function start(bench){
  if(!ready||!peerReady||run||dc?.readyState!=="open")throw Error("Both phones must be ready");
  const s={id:++serial,prompt:Array.from(enc.encode(bench?"Hello":$("prompt").value)),n:bench?128:Number($("maxBytes").value),temp:bench?0:Number($("temperature").value),bench};
  validateStart(s);control(T.START,s);begin(s);dispatch(s.prompt[0]);
}
const action=(id,fn)=>$(id).onclick=()=>Promise.resolve().then(fn).catch(fail);
action("host",()=>selectRole("a"));action("join",()=>selectRole("b"));
action("makeOffer",async()=>{makePeer();attach(pc.createDataChannel("hivegpt-v1",{ordered:true}));await pc.setLocalDescription(await pc.createOffer());await gather();status("Offer ready. Phone B imports it and returns an answer.");buttons();});
action("apply",()=>applyText($("remote").value));action("scan",startCamera);action("stopCamera",stopCamera);
action("copy",async()=>{try{await navigator.clipboard.writeText($("local").value);}catch(e){$("local").focus();$("local").select();status("Select and copy the pairing text manually.");}});
action("previous",()=>{frameIndex--;showFrame();});action("next",()=>{frameIndex++;showFrame();});action("inspect",inspect);
action("generate",()=>start(false));action("benchmark",()=>start(true));action("stop",()=>{if(run)run.stopping=true;});
$("temperature").oninput=()=>$("tempLabel").textContent=Number($("temperature").value).toFixed(1)+(Number($("temperature").value)===0?" · greedy":"");
if(new URLSearchParams(location.search).get("selftest")==="1"){
  $("roles").hidden=true;$("selfTest").hidden=false;status("TEST ONLY · full model diagnostic");
  action("testButton",()=>{
    $("testButton").disabled=true;$("testResult").textContent="Loading both halves…";const w=new Worker("worker.js");
    w.onerror=e=>{$("testResult").textContent=e.message;$("testButton").disabled=false;w.terminate();};
    w.onmessage=e=>{const m=e.data;if(m.type==="ready")w.postMessage({type:"test"});
      else if(m.type==="test"||m.type==="error"){$("testResult").textContent=JSON.stringify(m,null,2);$("testButton").disabled=false;w.terminate();}};
    w.postMessage({type:"load",role:"test"});
  });
}
if("serviceWorker" in navigator)navigator.serviceWorker.register("sw.js").catch(e=>status("Offline shell unavailable: "+e.message,true));
