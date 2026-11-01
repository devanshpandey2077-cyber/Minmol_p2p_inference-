"use strict";
const SHELL="split-gpt-shell-v2";
const ASSETS=["./","index.html","app.js","worker.js","qr.js"];
self.addEventListener("install",e=>e.waitUntil(caches.open(SHELL).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener("activate",e=>e.waitUntil(
  caches.keys().then(ks=>Promise.all(ks.filter(k=>k.startsWith("split-gpt-shell")&&k!==SHELL).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener("fetch",e=>{
  if(e.request.method!=="GET"||new URL(e.request.url).origin!==self.location.origin)return;
  const isShell=e.request.mode==="navigate"||/\.(js|html)$/.test(new URL(e.request.url).pathname);
  e.respondWith((async()=>{
    if(isShell){ // network first, cache as offline fallback
      try{const r=await fetch(e.request);const c=await caches.open(SHELL);c.put(e.request,r.clone());return r;}
      catch(_){return (await caches.match(e.request,{ignoreSearch:true}))||caches.match("index.html");}
    }
    return (await caches.match(e.request))||fetch(e.request);
  })());
});