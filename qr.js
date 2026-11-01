"use strict";
// Self-contained QR Model 2: version 5, level L, byte mode, mask 0.
// Scanner intentionally accepts this app's symbols, not arbitrary QR formats.
const QR = (() => {
  const N=37, DATA=108, ECC=26, exp=new Uint8Array(512), log=new Uint8Array(256);
  let a=1;
  for(let i=0;i<255;i++){exp[i]=a;log[a]=i;a<<=1;if(a&256)a^=0x11d;}
  for(let i=255;i<512;i++)exp[i]=exp[i-255];
  const mul=(a,b)=>a&&b?exp[log[a]+log[b]]:0;
  const div=(a,b)=>{if(!b)throw Error("GF division");return a?exp[(log[a]-log[b]+255)%255]:0;};
  function parity(data){
    let g=[1];
    for(let i=0;i<ECC;i++){
      const t=new Array(g.length+1).fill(0);
      for(let j=0;j<g.length;j++){t[j]^=g[j];t[j+1]^=mul(g[j],exp[i]);}g=t;
    }
    const r=new Uint8Array(DATA+ECC);r.set(data);
    for(let i=0;i<DATA;i++){const v=r[i];for(let j=0;j<g.length;j++)r[i+j]^=mul(v,g[j]);}
    return r.slice(DATA);
  }
  function syndromes(c){
    return Array.from({length:ECC},(_,k)=>{let v=0;for(const b of c)v=mul(v,exp[k])^b;return v;});
  }
  function correct(c){
    const s=syndromes(c);if(s.every(v=>v===0))return c;
    let C=[1],B=[1],L=0,m=1,b=1;
    for(let n=0;n<ECC;n++){
      let d=s[n];for(let i=1;i<=L;i++)d^=mul(C[i]||0,s[n-i]);
      if(!d){m++;continue;}
      const T=C.slice(),coef=div(d,b);
      while(C.length<B.length+m)C.push(0);
      for(let i=0;i<B.length;i++)C[i+m]^=mul(coef,B[i]);
      if(2*L<=n){L=n+1-L;B=T;b=d;m=1;}else m++;
    }
    if(L<1||L>13)throw Error("Uncorrectable QR");
    const positions=[];
    for(let p=0;p<c.length;p++){
      const x=exp[(255-(c.length-1-p)%255)%255];let v=0;
      for(let j=C.length-1;j>=0;j--)v=mul(v,x)^C[j];
      if(!v)positions.push(p);
    }
    if(positions.length!==L)throw Error("QR roots");
    // Solve syndrome equations for error magnitudes, avoiding a Forney convention dependency.
    const rows=Array.from({length:L},(_,k)=>[
      ...positions.map(p=>exp[(k*(c.length-1-p))%255]),s[k]]);
    for(let k=0;k<L;k++){
      let p=k;while(p<L&&!rows[p][k])p++;if(p===L)throw Error("QR singular");
      [rows[k],rows[p]]=[rows[p],rows[k]];
      const inv=div(1,rows[k][k]);for(let j=k;j<=L;j++)rows[k][j]=mul(rows[k][j],inv);
      for(let i=0;i<L;i++)if(i!==k){const v=rows[i][k];for(let j=k;j<=L;j++)rows[i][j]^=mul(v,rows[k][j]);}
    }
    positions.forEach((p,i)=>c[p]^=rows[i][L]);
    if(syndromes(c).some(Boolean))throw Error("QR checksum");return c;
  }
  function base(){
    const grid=Array.from({length:N},()=>new Uint8Array(N));
    const used=Array.from({length:N},()=>new Uint8Array(N));
    const set=(x,y,v)=>{if(x>=0&&y>=0&&x<N&&y<N){grid[y][x]=v?1:0;used[y][x]=1;}};
    for(let i=0;i<N;i++){set(6,i,i%2===0);set(i,6,i%2===0);}
    for(const [x,y] of [[3,3],[N-4,3],[3,N-4]])
      for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++){
        const d=Math.max(Math.abs(dx),Math.abs(dy));set(x+dx,y+dy,d!==2&&d!==4);
      }
    for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++)set(30+dx,30+dy,Math.max(Math.abs(dx),Math.abs(dy))!==1);
    const fmt=0x77c4, bit=i=>(fmt>>>i)&1;
    for(let i=0;i<=5;i++)set(8,i,bit(i));
    set(8,7,bit(6));set(8,8,bit(7));set(7,8,bit(8));
    for(let i=9;i<15;i++)set(14-i,8,bit(i));
    for(let i=0;i<8;i++)set(N-1-i,8,bit(i));
    for(let i=8;i<15;i++)set(8,N-15+i,bit(i));
    set(8,N-8,1);
    return {grid,used};
  }
  function traverse(used,callback){
    let up=true,index=0;
    for(let right=N-1;right>=1;right-=2){
      if(right===6)right=5;
      for(let v=0;v<N;v++){
        const y=up?N-1-v:v;
        for(let k=0;k<2;k++){const x=right-k;if(!used[y][x])callback(x,y,index++);}
      }up=!up;
    }
  }
  function encode(text){
    const bytes=new TextEncoder().encode(text);if(bytes.length>106)throw Error("QR frame too large");
    const bits=[],push=(v,n)=>{for(let i=n-1;i>=0;i--)bits.push((v>>>i)&1);};
    push(4,4);push(bytes.length,8);for(const b of bytes)push(b,8);
    for(let i=0;i<4&&bits.length<DATA*8;i++)bits.push(0);
    while(bits.length%8)bits.push(0);
    let pad=0;while(bits.length<DATA*8)push((pad++%2)?0x11:0xec,8);
    const data=new Uint8Array(DATA);
    bits.forEach((v,i)=>data[i>>3]|=v<<(7-(i&7)));
    const code=new Uint8Array(DATA+ECC);code.set(data);code.set(parity(data),DATA);
    const {grid,used}=base();
    traverse(used,(x,y,i)=>{const v=i<code.length*8?(code[i>>3]>>>(7-(i&7)))&1:0;grid[y][x]=v^((x+y)%2===0?1:0);});
    return grid;
  }
  function decodeGrid(grid){
    const {used}=base(),c=new Uint8Array(DATA+ECC);
    traverse(used,(x,y,i)=>{if(i<c.length*8)c[i>>3]|=(grid[y][x]^((x+y)%2===0?1:0))<<(7-(i&7));});
    correct(c);
    let bit=0;
    const get=n=>{let v=0;while(n--){v=(v<<1)|((c[bit>>3]>>>(7-(bit&7)))&1);bit++;}return v;};
    if(get(4)!==4)throw Error("QR mode");const n=get(8);if(n>106)throw Error("QR length");
    return new TextDecoder().decode(Uint8Array.from({length:n},()=>get(8)));
  }
  function solve(rows){
    const n=rows.length;
    for(let k=0;k<n;k++){
      let p=k;for(let i=k+1;i<n;i++)if(Math.abs(rows[i][k])>Math.abs(rows[p][k]))p=i;
      if(Math.abs(rows[p][k])<1e-10)throw Error("Degenerate QR");
      [rows[k],rows[p]]=[rows[p],rows[k]];const d=rows[k][k];for(let j=k;j<=n;j++)rows[k][j]/=d;
      for(let i=0;i<n;i++)if(i!==k){const f=rows[i][k];for(let j=k;j<=n;j++)rows[i][j]-=f*rows[k][j];}
    }return rows.map(r=>r[n]);
  }
  function homography(src,dst){
    const rows=[];
    src.forEach(([x,y],i)=>{const [u,v]=dst[i];rows.push([x,y,1,0,0,0,-u*x,-u*y,u],[0,0,0,x,y,1,-v*x,-v*y,v]);});
    const h=solve(rows);
    return (x,y)=>{const z=h[6]*x+h[7]*y+1;return [(h[0]*x+h[1]*y+h[2])/z,(h[3]*x+h[4]*y+h[5])/z];};
  }
  function scan(image){
    const {width:w,height:h,data}=image,hist=new Uint32Array(256),gray=new Uint8Array(w*h);
    for(let i=0;i<gray.length;i++){const j=i*4;gray[i]=(data[j]*77+data[j+1]*150+data[j+2]*29)>>8;hist[gray[i]]++;}
    let all=0;for(let i=0;i<256;i++)all+=i*hist[i];
    let count=0,sum=0,best=-1,threshold=128;
    for(let i=0;i<255;i++){
      count+=hist[i];sum+=i*hist[i];if(!count||count===w*h)continue;
      const d=sum/count-(all-sum)/(w*h-count),score=count*(w*h-count)*d*d;
      if(score>best){best=score;threshold=i;}
    }
    const pixel=(x,y)=>{x=Math.round(x);y=Math.round(y);return x>=0&&y>=0&&x<w&&y<h&&gray[y*w+x]<=threshold?1:0;};
    const ratio=c=>{const m=c.reduce((a,b)=>a+b,0)/7;return m>=1&&c.every((v,i)=>Math.abs(v-(i===2?3*m:m))<(i===2?1.7*m:0.85*m));};
    function cross(x,y,dx,dy){
      x=Math.round(x);y=Math.round(y);if(!pixel(x,y))return null;
      const c=[0,0,1,0,0];let t=1;
      while(t<Math.max(w,h)&&pixel(x-dx*t,y-dy*t)){c[2]++;t++;}
      const lo=t;
      for(let k=1;k>=0;k--)while(t<Math.max(w,h)&&pixel(x-dx*t,y-dy*t)===(k===0?1:0)){
        if(x-dx*t<0||y-dy*t<0||x-dx*t>=w||y-dy*t>=h)break;c[k]++;t++;
      }
      t=1;while(t<Math.max(w,h)&&pixel(x+dx*t,y+dy*t)){c[2]++;t++;}
      const hi=t;
      for(let k=3;k<=4;k++)while(t<Math.max(w,h)&&pixel(x+dx*t,y+dy*t)===(k===4?1:0)){
        if(x+dx*t<0||y+dy*t<0||x+dx*t>=w||y+dy*t>=h)break;c[k]++;t++;
      }
      return ratio(c)?{x:x+dx*(hi-lo)/2,y:y+dy*(hi-lo)/2,m:c.reduce((a,b)=>a+b,0)/7}:null;
    }
    const candidates=[];
    for(let y=0;y<h;y++){
      const runs=[];let start=0,color=pixel(0,y);
      for(let x=1;x<=w;x++)if(x===w||pixel(x,y)!==color){
        runs.push({start,len:x-start,color});start=x;color=1-color;
        if(runs.length>=5){const r=runs.slice(-5),c=r.map(r=>r.len);
          if(r[0].color&&ratio(c)){
            const cx=r[2].start+(r[2].len-1)/2,v=cross(cx,y,0,1),u=v&&cross(cx,v.y,1,0);
            if(u){const old=candidates.find(p=>Math.hypot(p.x-u.x,p.y-u.y)<u.m*2);
              if(old){old.x=(old.x*old.count+u.x)/(old.count+1);old.y=(old.y*old.count+u.y)/(old.count+1);old.count++;}
              else candidates.push({...u,count:1});
            }
          }
        }
      }
    }
    candidates.sort((a,b)=>b.count-a.count);const c=candidates.slice(0,12),dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
    for(let i=0;i<c.length;i++)for(let j=i+1;j<c.length;j++)for(let k=j+1;k<c.length;k++){
      const t=[c[i],c[j],c[k]];
      for(let corner=0;corner<3;corner++){
        const tl=t[corner];let tr=t[(corner+1)%3],bl=t[(corner+2)%3];
        const ab=dist(tl,tr),ac=dist(tl,bl),bc=dist(tr,bl);
        if(ab<30||ac<30||ab/ac<0.5||ab/ac>2||Math.abs(ab*ab+ac*ac-bc*bc)>ab*ac*0.65)continue;
        if((tr.x-tl.x)*(bl.y-tl.y)-(tr.y-tl.y)*(bl.x-tl.x)<0)[tr,bl]=[bl,tr];
        const ux=(tr.x-tl.x)/30,uy=(tr.y-tl.y)/30,vx=(bl.x-tl.x)/30,vy=(bl.y-tl.y)/30;
        const ex=tl.x+27*(ux+vx),ey=tl.y+27*(uy+vy),module=(ab+ac)/60;
        let align=null,lowest=Infinity;
        for(let oy=-5;oy<=5;oy+=0.5)for(let ox=-5;ox<=5;ox+=0.5){
          const px=ex+ox*module,py=ey+oy*module;let mismatch=0;
          for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++)
            if(pixel(px+dx*ux+dy*vx,py+dx*uy+dy*vy)!==(Math.max(Math.abs(dx),Math.abs(dy))!==1?1:0))mismatch++;
          const score=mismatch+0.001*(ox*ox+oy*oy);
          if(score<lowest){lowest=score;align=[px,py];}
        }
        const attempts=[null];if(lowest<5)attempts.unshift(align);
        for(const alignment of attempts)try{
          const map=alignment?homography([[3.5,3.5],[33.5,3.5],[3.5,33.5],[30.5,30.5]],
            [[tl.x,tl.y],[tr.x,tr.y],[bl.x,bl.y],alignment]):
            (x,y)=>[tl.x+(x-3.5)*ux+(y-3.5)*vx,tl.y+(x-3.5)*uy+(y-3.5)*vy];
          const grid=Array.from({length:N},(_,y)=>Uint8Array.from({length:N},(_,x)=>pixel(...map(x+0.5,y+0.5))));
          const text=decodeGrid(grid);if(text.startsWith("HG1:"))return text;
        }catch(e){/* Try the next geometric hypothesis. */}
      }
    }
    return null;
  }
  function draw(canvas,text){
    const g=encode(text),scale=8;canvas.width=canvas.height=(N+8)*scale;
    const ctx=canvas.getContext("2d");ctx.fillStyle="white";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle="black";
    for(let y=0;y<N;y++)for(let x=0;x<N;x++)if(g[y][x])ctx.fillRect((x+4)*scale,(y+4)*scale,scale,scale);
  }
  return {encode,decodeGrid,scan,draw};
})();
if(typeof window!=="undefined")window.QR=QR;
else if(typeof self!=="undefined")self.onmessage=e=>{
  try{self.postMessage({text:QR.scan(e.data)});}catch(e){self.postMessage({text:null});}
};
