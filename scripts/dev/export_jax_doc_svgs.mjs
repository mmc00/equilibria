#!/usr/bin/env node
// Writes the figures of docs/site/guide/jax_backend.md as SVG files.
//
//   node scripts/dev/export_jax_doc_svgs.mjs docs/assets/jax-backend
//
// The drawing kit comes from the equilibria Lego mockup (2026-11-12): oblique
// projection with U=26 (stud pitch), H=24 (brick height), K=0.36 (depth). Every
// colour is a CSS variable or a class from docs/site/_static/jax_backend.css, so
// the inlined SVGs follow the site's light/dark toggle. One concept is always the
// same piece and colour: see the "parts" figure.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/* ---------- Lego drawing kit (from the equilibria Lego mockup, 2026-11-12) ---------- */
const U=26, H=24, K=0.36; // stud pitch, brick height, oblique depth factor
const P=(x,y,z)=>[x+K*y, -z-K*y];
const pt=a=>a.map(v=>v.toFixed(1)).join(",");
const poly=(pts,fill)=>`<polygon points="${pts.map(pt).join(" ")}" style="fill:${fill}"/>`;
const top=c=>`color-mix(in oklab, ${c} 74%, white)`;
const side=c=>`color-mix(in oklab, ${c} 70%, black)`;
const studSide=c=>`color-mix(in oklab, ${c} 82%, black)`;
const T=(x,y,b)=>`<g transform="translate(${x},${y})">${b}</g>`;
const svg=(vb,body,cls="")=>`<svg viewBox="${vb}" role="img" xmlns="http://www.w3.org/2000/svg" class="jxb ${cls}">${body}</svg>`;
const ARR=`<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="head"/></marker><marker id="ahj" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="head-jax"/></marker><marker id="ahp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="head-pyo"/></marker></defs>`;
const FLAMEDEFS=`<defs><linearGradient id="fl-out" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:var(--flame-3)"/><stop offset=".55" style="stop-color:var(--flame-4)"/><stop offset="1" style="stop-color:var(--flame-5);stop-opacity:0"/></linearGradient><linearGradient id="fl-in" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:var(--flame-1)"/><stop offset=".6" style="stop-color:var(--flame-2)"/><stop offset="1" style="stop-color:var(--flame-3);stop-opacity:0"/></linearGradient><radialGradient id="smoke" cx=".5" cy=".5" r=".5"><stop offset="0" style="stop-color:var(--smoke);stop-opacity:.9"/><stop offset="1" style="stop-color:var(--smoke);stop-opacity:0"/></radialGradient></defs>`;

function studs(x,y,z,nx,ny,c){let s="";const r=U*.3,sh=U*.2;
  for(let j=ny-1;j>=0;j--)for(let i=0;i<nx;i++){const [cx,cy]=P(x+(i+.5)*U,y+(j+.5)*U,z);
    s+=`<path d="M${cx-r},${cy} L${cx-r},${cy-sh} A${r},${r*.42} 0 0 1 ${cx+r},${cy-sh} L${cx+r},${cy} A${r},${r*.42} 0 0 1 ${cx-r},${cy}Z" style="fill:${studSide(c)}"/>`;
    s+=`<ellipse cx="${cx}" cy="${cy-sh}" rx="${r}" ry="${r*.42}" style="fill:${top(c)}"/>`;}
  return s;}
function brick(x,y,z,nx,ny,h,c,o={}){const w=nx*U,d=ny*U;let s="";
  s+=poly([P(x,y,z+h),P(x+w,y,z+h),P(x+w,y+d,z+h),P(x,y+d,z+h)],top(c));
  s+=poly([P(x+w,y,z),P(x+w,y+d,z),P(x+w,y+d,z+h),P(x+w,y,z+h)],side(c));
  s+=poly([P(x,y,z),P(x+w,y,z),P(x+w,y,z+h),P(x,y,z+h)],c);
  if(o.studs!==false)s+=studs(x,y,z+h,nx,ny,c);
  if(o.label){const [lx,ly]=P(x+w/2,y,z+h/2);s+=`<text x="${lx}" y="${ly+4.5}" text-anchor="middle" class="on-brick" style="font-size:${o.fs||12.5}px">${o.label}</text>`;}
  return s;}
const plate=(x,y,z,nx,ny,c)=>brick(x,y,z,nx,ny,8,c);
function tile(x,y,z,nx,ny,c,o={}){let s=brick(x,y,z,nx,ny,o.h||14,c,{studs:false,label:o.label,fs:o.fs});
  const [a1,a2]=P(x,y,z+2),[b1,b2]=P(x+nx*U,y,z+2);
  s+=`<line x1="${a1}" y1="${a2}" x2="${b1}" y2="${b2}" style="stroke:color-mix(in oklab, ${c} 60%, black);stroke-width:1"/>`;return s;}
function sheet(x,y,w,h,rows){
  let o=`<path d="M${x},${y} H${x+w-16} L${x+w},${y+16} V${y+h} H${x}Z" style="fill:var(--jxb-chip);stroke:var(--jxb-line);stroke-width:1.2"/><path d="M${x+w-16},${y} V${y+16} H${x+w}" style="fill:var(--jxb-line)"/>`;
  o+=`<text x="${x+12}" y="${y+22}" class="disp" style="font-size:13px;font-weight:600">GTAP</text>`;
  rows.forEach((r,i)=>{const yy=y+34+i*15;o+=`<rect x="${x+12}" y="${yy}" width="16" height="9" rx="2" style="fill:${r[0]}"/><rect x="${x+34}" y="${yy+3}" width="${r[1]}" height="3" rx="1.5" style="fill:var(--jxb-muted);opacity:.55"/>`;});
  return o;}
function flight(x,y,w){const st=[["base","not solved"],["check","solved"],["shock","solved"]];
  let o=`<rect x="${x}" y="${y}" width="${w}" height="150" rx="10" style="fill:var(--jxb-chip);stroke:var(--jxb-line);stroke-width:1.2"/><path d="M${x+28},${y+34} V${y+124}" style="stroke:var(--pyo);stroke-width:2.5"/>`;
  st.forEach((s,i)=>{const yy=y+34+i*45;
    o+=`<circle cx="${x+28}" cy="${yy}" r="8" style="fill:${i==2?'var(--c-shock)':'var(--pyo)'}"/>`;
    o+=`<text x="${x+44}" y="${yy+1}" class="disp" style="font-size:14px;font-weight:600">${s[0]}</text><text x="${x+44}" y="${yy+15}" class="mono sm">${s[1]}</text>`;});
  return o;}
function pilot(cx,by,s=1){const q=v=>v*s;let o="";
  o+=`<rect x="${cx-q(15)}" y="${by-q(34)}" width="${q(14)}" height="${q(34)}" rx="${q(2)}" style="fill:var(--suit)"/><rect x="${cx+q(1)}" y="${by-q(34)}" width="${q(14)}" height="${q(34)}" rx="${q(2)}" style="fill:var(--suit)"/>`;
  o+=`<rect x="${cx-q(16)}" y="${by-q(42)}" width="${q(32)}" height="${q(9)}" rx="${q(2)}" style="fill:var(--suit-dk)"/>`;
  o+=`<path d="M${cx-q(16)},${by-q(42)} L${cx-q(13)},${by-q(74)} L${cx+q(13)},${by-q(74)} L${cx+q(16)},${by-q(42)}Z" style="fill:var(--c-shock)"/>`;
  o+=`<path d="M${cx-q(6)},${by-q(74)} L${cx},${by-q(64)} L${cx+q(6)},${by-q(74)}" style="fill:none;stroke:var(--brick-ink);stroke-width:${q(2)}"/>`;
  for(const sx of[-1,1]){o+=`<path d="M${cx+sx*q(13)},${by-q(72)} L${cx+sx*q(22)},${by-q(52)}" style="stroke:var(--c-shock);stroke-width:${q(8)};stroke-linecap:round"/><circle cx="${cx+sx*q(23)}" cy="${by-q(47)}" r="${q(4.5)}" style="fill:var(--skin)"/>`;}
  o+=`<rect x="${cx-q(4)}" y="${by-q(79)}" width="${q(8)}" height="${q(6)}" style="fill:var(--skin)"/><rect x="${cx-q(11)}" y="${by-q(100)}" width="${q(22)}" height="${q(22)}" rx="${q(6)}" style="fill:var(--skin)"/>`;
  o+=`<path d="M${cx-q(13)},${by-q(90)} Q${cx-q(13)},${by-q(108)} ${cx},${by-q(108)} Q${cx+q(13)},${by-q(108)} ${cx+q(13)},${by-q(90)}Z" style="fill:var(--helmet);stroke:var(--helmet-line);stroke-width:1"/>`;
  o+=`<rect x="${cx-q(10)}" y="${by-q(97)}" width="${q(20)}" height="${q(4)}" rx="${q(2)}" style="fill:var(--visor);opacity:.85"/>`;
  o+=`<circle cx="${cx-q(4)}" cy="${by-q(88)}" r="${q(1.4)}" style="fill:var(--face)"/><circle cx="${cx+q(4)}" cy="${by-q(88)}" r="${q(1.4)}" style="fill:var(--face)"/><path d="M${cx-q(4)},${by-q(84)} Q${cx},${by-q(81)} ${cx+q(4)},${by-q(84)}" style="fill:none;stroke:var(--face);stroke-width:${q(1.2)}"/>`;
  return o;}
function flame(cx,cy,L,w){return `<path d="M${cx-w},${cy} C${cx-w},${cy+L*.35} ${cx-w*.35},${cy+L*.7} ${cx},${cy+L} C${cx+w*.35},${cy+L*.7} ${cx+w},${cy+L*.35} ${cx+w},${cy}Z" fill="url(#fl-out)"/><path d="M${cx-w*.55},${cy} C${cx-w*.55},${cy+L*.3} ${cx-w*.2},${cy+L*.5} ${cx},${cy+L*.62} C${cx+w*.2},${cy+L*.5} ${cx+w*.55},${cy+L*.3} ${cx+w*.55},${cy}Z" fill="url(#fl-in)"/>`;}
function nozzle(cx,cy,w,h){return `<path d="M${cx-w*.35},${cy} L${cx+w*.35},${cy} L${cx+w*.6},${cy+h} L${cx-w*.6},${cy+h}Z" style="fill:var(--c-engine)"/><path d="M${cx-w*.35},${cy} L${cx-w*.05},${cy} L${cx-w*.2},${cy+h} L${cx-w*.6},${cy+h}Z" style="fill:var(--c-engine-lt)"/><ellipse cx="${cx}" cy="${cy+h}" rx="${w*.6}" ry="${w*.16}" style="fill:var(--c-engine-dk)"/>`;}
const FAM=[["Shock","--c-shock"],["Export (CET)","--c-trade"],["Production","--c-prod"],["Factors","--c-factor"],["Import","--c-trade"],["Households","--c-hh"],["Income, gov.","--c-gov"],["Closure","--c-closure"]];
function rocket(cx,base,fl=1,small=false){const nx=2,w=nx*U,hh=small?14:18;let o="";const x=cx-w/2-K*U/2;
  o+=plate(x-U*1.5,0,0,nx+3,2,"var(--c-data)");let z=8;const lift=fl?70:0;
  if(fl){const [fx,fy]=P(x+w/2,U,z+lift);o+=`<ellipse cx="${fx-30}" cy="${fy+lift-6}" rx="26" ry="12" fill="url(#smoke)"/><ellipse cx="${fx+34}" cy="${fy+lift-4}" rx="30" ry="13" fill="url(#smoke)"/>`+nozzle(fx,fy,34,12)+flame(fx,fy+12,lift-6,16);}
  z+=lift;const [f1x,f1y]=P(x,U,z),[f2x,f2y]=P(x+w,U,z);
  o+=`<path d="M${f1x},${f1y-30} L${f1x-22},${f1y+4} L${f1x},${f1y-4}Z" style="fill:${side('var(--c-shock)')}"/>`;
  FAM.forEach((f,i)=>{o+=brick(x,0,z,nx,2,hh,`var(${f[1]})`,{studs:i==FAM.length-1});z+=hh;});
  o+=`<path d="M${f2x},${f2y-30} L${f2x+22},${f2y+4} L${f2x},${f2y-4}Z" style="fill:var(--c-shock)"/>`;
  const a=P(x,0,z),b=P(x+w,0,z),c=P(x+w,2*U,z),tip=P(x+w/2,U,z+46);
  o+=poly([a,b,tip],"var(--c-closure)")+poly([b,c,tip],side("var(--c-closure)"));
  return T(0,base,o);}

/* ---------- new pieces for the JAX route ---------- */
// clay: a soft slab with no studs
function clay(x,y,z,nx,ny,h=16){const w=nx*U,d=ny*U,c="var(--c-clay)";let s="";
  s+=poly([P(x,y,z+h),P(x+w,y,z+h),P(x+w,y+d,z+h),P(x,y+d,z+h)],top(c));
  s+=poly([P(x+w,y,z),P(x+w,y+d,z),P(x+w,y+d,z+h),P(x+w,y,z+h)],side(c));
  s+=poly([P(x,y,z),P(x+w,y,z),P(x+w,y,z+h),P(x,y,z+h)],c);
  for(let i=1;i<4;i++){const [a1,a2]=P(x+w*i/4-6,y,z+h*.55),[b1,b2]=P(x+w*i/4+6,y,z+h*.45);s+=`<path d="M${a1},${a2} Q${(a1+b1)/2},${a2-3} ${b1},${b2}" style="fill:none;stroke:color-mix(in oklab, ${c} 75%, black);stroke-width:1;opacity:.6"/>`;}
  return s;}
// mold: clay with the imprint of a brick (stud holes and its outline)
function mold(x,y,z,nx,ny,o={}){const h=16;let s=clay(x,y,z,nx+1,ny+1,h);const ox=x+U*.5,oy=y+U*.5,zt=z+h;
  s+=poly([P(ox,oy,zt),P(ox+nx*U,oy,zt),P(ox+nx*U,oy+ny*U,zt),P(ox,oy+ny*U,zt)],"color-mix(in oklab, var(--c-clay) 62%, black)");
  for(let j=ny-1;j>=0;j--)for(let i=0;i<nx;i++){const [cx,cy]=P(ox+(i+.5)*U,oy+(j+.5)*U,zt);s+=`<ellipse cx="${cx}" cy="${cy}" rx="${U*.3}" ry="${U*.3*.42}" style="fill:var(--c-clay-hole)"/>`;}
  if(o.label){const [lx,ly]=P(x+(nx+1)*U/2,y,z+h/2);s+=`<text x="${lx}" y="${ly+4}" text-anchor="middle" class="on-brick" style="font-size:${o.fs||11}px">${o.label}</text>`;}
  return s;}
// storage box: open cardboard box with labelled drawers
function storageBox(x,y,w,h,labels){const c="var(--c-box)",d=46;let s="";
  s+=`<path d="M${x},${y} L${x+d*.9},${y-d*.45} L${x+w+d*.9},${y-d*.45} L${x+w},${y}Z" style="fill:color-mix(in oklab, ${c} 55%, black)"/>`;
  s+=`<path d="M${x+w},${y} L${x+w+d*.9},${y-d*.45} V${y+h-d*.45} L${x+w},${y+h}Z" style="fill:${side(c)}"/>`;
  s+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" style="fill:${c}"/>`;
  s+=`<path d="M${x},${y} L${x-14},${y-20} L${x+w-14},${y-20} L${x+w},${y}Z" style="fill:${top(c)}"/>`;
  labels.forEach((l,i)=>{const yy=y+16+i*30;s+=`<rect x="${x+14}" y="${yy}" width="${w-28}" height="22" rx="4" style="fill:var(--jxb-surface);stroke:color-mix(in oklab, ${c} 60%, black);stroke-width:1"/><text x="${x+24}" y="${yy+15}" class="mono" style="font-size:10.5px">${l}</text>`;});
  return s;}
// box photo: the front of a set box with the finished model printed on it
function boxPhoto(x,y,w,h){let s=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" style="fill:var(--c-closure)"/><rect x="${x+10}" y="${y+30}" width="${w-20}" height="${h-44}" rx="4" style="fill:var(--c-photo)"/>`;
  s+=`<text x="${x+12}" y="${y+21}" class="on-brick" style="font-size:13px">GTAP · box photo</text>`;
  s+=T(x+w/2-58,y+h-14,`<g transform="scale(.5)">${rocket(110,0,0,true).replace(/^<g transform="translate\(0,0\)">/,'<g>')}</g>`);
  return s;}

/* ---------- the figures (each returns a complete <svg>) ---------- */
const PARTS=[
 ["Brick","Block","One block: households, production, trade…",()=>svg("0 0 150 110",T(40,96,brick(0,0,0,3,2,24,"var(--c-prod)")))],
 ["Plate","data (params)","The SAM and the parameters: everything rests on it",()=>svg("0 0 150 110",T(26,92,plate(0,0,0,4,3,"var(--c-data)")))],
 ["Smooth tile","ClosureBlock","No studs, nothing goes on top: it closes the model",()=>svg("0 0 150 110",T(30,90,tile(0,0,0,3,2,"var(--c-closure)",{h:18,label:"Closure",fs:12})))],
 ["Instruction sheet","Template","Which bricks the model has and how they fit",()=>svg("0 0 150 110",sheet(36,8,78,96,[["var(--c-shock)",30],["var(--c-trade)",24],["var(--c-prod)",32],["var(--c-hh)",22]]))],
 ["Flight plan","Recipe","What we do with the rocket: base, check, shock",()=>svg("0 0 150 170",flight(10,10,130))],
 ["Pilot","Driver","Flies the plan step by step, decides nothing",()=>svg("0 0 150 112",pilot(75,108,.95))],
 ["Rocket","Exercise","The mission: this rocket, this plan, this shock",()=>svg("10 60 130 260",FLAMEDEFS+rocket(75,300,.8,true))],
 ["Clay","RecordingModel","Takes the shape of whatever is pressed into it",()=>svg("0 0 150 110",T(24,88,clay(0,0,0,4,2,20))),1],
 ["Mold","a recorded tree","The shape of one equation, kept to make copies",()=>svg("0 0 150 110",T(20,90,mold(0,0,0,3,1))),1],
 ["Storage box","disk caches","Where molds and compiled programs wait for the next run",()=>svg("0 0 150 120",storageBox(22,34,88,76,["records","jax_aot"])),1],
 ["Box photo","Pyomo photo","The finished model, saved, to compare against",()=>svg("0 0 150 140",boxPhoto(20,8,110,124)),1],
 ["Engine","Newton + MUMPS","Takes F and J from either route and solves",()=>svg("0 0 150 120",FLAMEDEFS+T(36,70,brick(0,0,0,3,2,22,"var(--c-engine)",{studs:true}))+nozzle(64,70,30,14)+flame(64,84,30,13)),1]
];
const FIGS={
 stack:()=>{let o=plate(0,0,0,8,2,"var(--c-data)");const B=[["Production","--c-prod"],["Factors","--c-factor"],["Trade","--c-trade"],["Households","--c-hh"],["Government","--c-gov"]];B.forEach((b,i)=>{o+=brick((i%2)*U*.6,0,8+i*26,7,2,26,`var(${b[1]})`,{studs:false,label:b[0],fs:12});});o+=tile(U*.6,0,8+5*26,7,2,"var(--c-closure)",{h:16,label:"Closure",fs:11});return svg("0 0 290 230",T(14,212,o));},
 parts:()=>{let o=FLAMEDEFS;PARTS.forEach((p,k)=>{const c=k%4,r=Math.floor(k/4),x=c*240,y=r*230;const inner=p[3]().replace(/<defs>.*?<\/defs>/g,"").replace(/^<svg /,`<svg x="${x+20}" y="${y+10}" width="200" height="140" `);
   o+=`<rect x="${x+4}" y="${y+2}" width="232" height="220" rx="12" class="box" ${p[4]?'style="stroke-dasharray:5 4"':''}/>`+inner+`<text x="${x+20}" y="${y+174}" class="disp" style="font-size:16px;font-weight:600">${p[0]}</text><text x="${x+20}" y="${y+194}" class="mono sm">${p[1]}</text>`;
   if(p[4])o+=`<text x="${x+222}" y="${y+24}" text-anchor="end" class="mono tj" style="font-size:10px">JAX ROUTE</text>`;});
  return svg("0 0 960 690",o,"wide");},
 big_picture:()=>svg("0 0 960 430",ARR+`
  <rect x="330" y="14" width="300" height="66" rx="10" class="box-common"/><text x="480" y="42" text-anchor="middle" class="lbl" font-weight="700">GTAP blocks (the bricks)</text><text x="480" y="62" text-anchor="middle" class="sm">rules: build_expression(m, indices)</text>
  <path d="M420,80 C420,110 230,105 230,132" class="ln-pyo" marker-end="url(#ahp)"/><path d="M540,80 C540,110 730,105 730,132" class="ln-jax" marker-end="url(#ahj)"/>
  <text x="250" y="112" class="tp lbl" font-weight="700">Pyomo route</text><text x="640" y="112" class="tj lbl" font-weight="700">JAX route (family)</text>
  <rect x="110" y="136" width="240" height="56" rx="8" class="box-pyo"/><text x="230" y="160" text-anchor="middle" class="lbl">Pyomo model</text><text x="230" y="178" text-anchor="middle" class="sm">one object per cell</text>
  <path d="M230,192 L230,222" class="ln-pyo" marker-end="url(#ahp)"/>
  <rect x="110" y="226" width="240" height="56" rx="8" class="box-pyo"/><text x="230" y="250" text-anchor="middle" class="lbl">pynumero (ASL, in C)</text><text x="230" y="268" text-anchor="middle" class="sm">reads the .nl file and evaluates</text>
  <rect x="610" y="136" width="240" height="56" rx="8" class="box-jax"/><text x="730" y="160" text-anchor="middle" class="lbl">Recorder → molds (trees)</text><text x="730" y="178" text-anchor="middle" class="sm">once, kept in the storage box</text>
  <path d="M730,192 L730,222" class="ln-jax" marker-end="url(#ahj)"/>
  <rect x="610" y="226" width="240" height="56" rx="8" class="box-jax"/><text x="730" y="250" text-anchor="middle" class="lbl">JAX molds by shape (families)</text><text x="730" y="268" text-anchor="middle" class="sm">vmap + jacrev, compiled</text>
  <path d="M230,282 C230,318 400,312 420,334" class="ln-pyo" marker-end="url(#ahp)"/><path d="M730,282 C730,318 560,312 540,334" class="ln-jax" marker-end="url(#ahj)"/>
  <text x="480" y="322" text-anchor="middle" class="lbl mono">F(x) and J(x)</text>
  <rect x="330" y="338" width="300" height="66" rx="10" class="box-common"/><text x="480" y="366" text-anchor="middle" class="lbl" font-weight="700">The engine: Newton with a trust region</text><text x="480" y="386" text-anchor="middle" class="sm">MUMPS factors J · solver/newton_tr.py</text>`,"wide"),
 same_brick:()=>svg("0 0 340 230",ARR+T(118,104,brick(0,0,0,4,2,26,"var(--f-ces)",{label:"eq_xaa_activity",fs:11.5}))+`
  <path d="M150,110 C120,150 80,150 70,176" class="ln-pyo" marker-end="url(#ahp)"/><path d="M190,110 C230,150 260,150 268,176" class="ln-jax" marker-end="url(#ahj)"/>
  <rect x="10" y="180" width="120" height="40" rx="8" class="box-pyo"/><text x="70" y="205" text-anchor="middle" class="lbl">Pyomo model</text>
  <rect x="210" y="180" width="120" height="40" rx="8" class="box-jax"/><text x="270" y="205" text-anchor="middle" class="lbl">RecordingModel</text>`),
 two_objects:()=>svg("0 0 960 300",ARR+`
  <rect x="360" y="100" width="240" height="100" rx="12" class="box-common"/><text x="480" y="132" text-anchor="middle" class="lbl" font-weight="700">eq_xaa_activity</text><text x="480" y="154" text-anchor="middle" class="sm mono">build_expression(m, (r,i,a))</text><text x="480" y="176" text-anchor="middle" class="sm">asks for: m.xaa · m.nd · m.pnd · m.pa</text>
  <rect x="20" y="40" width="250" height="70" rx="8" class="box-pyo"/><text x="145" y="68" text-anchor="middle" class="lbl">m = Pyomo model</text><text x="145" y="88" text-anchor="middle" class="sm">ConcreteModel</text><path d="M270,80 C320,80 320,125 356,128" class="ln-pyo" marker-end="url(#ahp)"/>
  <rect x="20" y="190" width="250" height="70" rx="8" class="box-jax"/><text x="145" y="218" text-anchor="middle" class="lbl">m = RecordingModel (the clay)</text><text x="145" y="238" text-anchor="middle" class="sm">family/recorder.py</text><path d="M270,225 C320,225 320,175 356,172" class="ln-jax" marker-end="url(#ahj)"/>
  <path d="M600,128 C640,128 640,80 686,80" class="ln-pyo" marker-end="url(#ahp)"/><rect x="690" y="40" width="250" height="70" rx="8" class="box-pyo"/><text x="815" y="68" text-anchor="middle" class="lbl">a Pyomo constraint</text><text x="815" y="88" text-anchor="middle" class="sm">goes to Constraint, then to the .nl</text>
  <path d="M600,172 C640,172 640,225 686,225" class="ln-jax" marker-end="url(#ahj)"/><rect x="690" y="190" width="250" height="70" rx="8" class="box-jax"/><text x="815" y="218" text-anchor="middle" class="lbl">a tree, Residual (the mold)</text><text x="815" y="238" text-anchor="middle" class="sm">goes to FamilyBackend.records</text>`,"wide"),
 press:()=>svg("0 0 380 250",ARR+T(30,70,brick(0,0,0,3,2,24,"var(--f-ces)",{label:"rule",fs:12}))+`<path d="M78,80 L78,128" class="ln dash" marker-end="url(#ah)"/><text x="88" y="108" class="sm">press</text>`+T(16,214,clay(0,0,0,4,3,22))+`<text x="74" y="244" text-anchor="middle" class="sm">clay: RecordingModel</text>
  <path d="M170,180 L222,180" class="ln-jax" marker-end="url(#ahj)"/>`+T(232,206,mold(0,0,0,3,2,{label:"mold"}))+`<text x="300" y="244" text-anchor="middle" class="sm">mold: one tree per cell</text>`),
 tree:()=>svg("0 0 960 360",`<g class="ln"><path d="M480,52 L230,112"/><path d="M480,52 L700,112"/><path d="M700,140 L560,200"/><path d="M700,140 L700,200"/><path d="M700,140 L860,200"/><path d="M560,228 L470,290"/><path d="M560,228 L640,290"/><path d="M470,318 L400,318"/></g>
  <rect x="390" y="20" width="180" height="34" rx="17" class="node-op"/><text x="480" y="42" text-anchor="middle" class="lbl mono">Residual (left − right)</text>
  <rect x="150" y="112" width="160" height="32" rx="6" class="node-var"/><text x="230" y="133" text-anchor="middle" class="lbl mono">xaa[USA,Food,agr]</text><text x="230" y="162" text-anchor="middle" class="sm">left side: one variable</text>
  <rect x="640" y="110" width="120" height="32" rx="16" class="node-op"/><text x="700" y="131" text-anchor="middle" class="lbl mono">mul</text>
  <rect x="480" y="200" width="160" height="32" rx="16" class="node-op"/><text x="560" y="221" text-anchor="middle" class="lbl mono">pow</text>
  <rect x="650" y="200" width="100" height="32" rx="6" class="node-var"/><text x="700" y="221" text-anchor="middle" class="lbl mono">nd[USA,agr]</text>
  <rect x="790" y="200" width="140" height="32" rx="6" class="node-cst"/><text x="860" y="221" text-anchor="middle" class="lbl mono">0.0312</text><text x="860" y="250" text-anchor="middle" class="sm">io · λ^(σ−1), a number</text>
  <rect x="400" y="290" width="140" height="32" rx="16" class="node-op"/><text x="470" y="311" text-anchor="middle" class="lbl mono">div</text>
  <rect x="590" y="290" width="100" height="32" rx="6" class="node-cst"/><text x="640" y="311" text-anchor="middle" class="lbl mono">0.5</text><text x="640" y="340" text-anchor="middle" class="sm">σ, a number</text>
  <text x="150" y="300" class="sm">div has two leaves:</text>
  <rect x="150" y="306" width="110" height="28" rx="6" class="node-var"/><text x="205" y="325" text-anchor="middle" class="sm mono ink">pnd[USA,agr]</text>
  <rect x="268" y="306" width="122" height="28" rx="6" class="node-var"/><text x="329" y="325" text-anchor="middle" class="sm mono ink">pa[USA,Food,agr]</text>`,"wide"),
 one_mold:()=>{let o=ARR+T(20,120,mold(0,0,0,3,2,{label:"one mold"}))+`<path d="M150,96 L196,96" class="ln-jax" marker-end="url(#ahj)"/><text x="173" y="86" text-anchor="middle" class="tj mono lbl">vmap</text>`;
   const cols=["var(--f-ces)"];for(let r=0;r<2;r++)for(let i=0;i<3;i++)o+=T(206+i*58+r*14,78+r*44,brick(0,0,0,1,1,16,cols[0],{}));
   o+=`<text x="300" y="168" text-anchor="middle" class="sm">… a thousand equal pieces</text>`;return svg("0 0 400 180",o);},
 families:()=>svg("0 0 960 330",ARR+`<text x="20" y="26" class="lbl" font-weight="700">Recorded cells</text>
  <rect x="20" y="40" width="250" height="40" rx="6" class="box"/><text x="32" y="65" class="sm mono ink">xaa[USA,Food,agr] − 0.031·nd·(pnd/pa)^0.5</text>
  <rect x="20" y="88" width="250" height="40" rx="6" class="box"/><text x="32" y="113" class="sm mono ink">xaa[CHN,Mfg,mfg] − 0.207·nd·(pnd/pa)^0.0</text>
  <rect x="20" y="136" width="250" height="40" rx="6" class="box"/><text x="32" y="161" class="sm mono ink">xaa[ROW,Svc,svc] − 0.118·nd·(pnd/pa)^0.5</text>
  <text x="145" y="198" text-anchor="middle" class="sm">… thousands more, same shape</text>
  <path d="M278,108 L346,108" class="ln-jax" marker-end="url(#ahj)"/>
  <text x="355" y="26" class="lbl" font-weight="700">One mold</text><rect x="355" y="40" width="240" height="88" rx="10" class="box-jax"/>
  <text x="475" y="72" text-anchor="middle" class="lbl mono">f(x₀,x₁,x₂,x₃; c₀,c₁) =</text><text x="475" y="96" text-anchor="middle" class="lbl mono">x₀ − c₀·x₁·(x₂/x₃)^c₁</text><text x="475" y="118" text-anchor="middle" class="sm">a small JAX function</text>
  <text x="355" y="160" class="lbl" font-weight="700">+ a table</text><rect x="355" y="172" width="240" height="116" rx="8" class="box"/>
  <text x="370" y="196" class="sm mono">row   x columns          c</text><text x="370" y="218" class="sm mono ink">0     37  912 1504 2210  .031 .5</text><text x="370" y="240" class="sm mono ink">1     58  933 1525 2231  .207 0</text><text x="370" y="262" class="sm mono ink">2     79  954 1546 2252  .118 .5</text><text x="370" y="281" class="sm mono">…</text>
  <path d="M603,160 L671,160" class="ln-jax" marker-end="url(#ahj)"/>
  <text x="680" y="26" class="lbl" font-weight="700">JAX computes all of it at once</text><rect x="680" y="40" width="260" height="110" rx="10" class="box-common"/>
  <text x="700" y="72" class="lbl mono">vmap(f)</text><text x="784" y="72" class="sm">→ F for every row</text><text x="700" y="104" class="lbl mono">vmap(jacrev(f))</text><text x="700" y="126" class="sm">→ exact derivatives, for J</text>
  <rect x="680" y="170" width="260" height="118" rx="10" class="box"/><text x="700" y="196" class="sm ink">The table does not change between</text><text x="700" y="216" class="sm ink">iterations, so neither does where J</text><text x="700" y="236" class="sm ink">has zeros. MUMPS prepares its analysis</text><text x="700" y="256" class="sm ink">once and reuses it.</text>`,"wide"),
 engine:()=>svg("0 0 380 250",ARR+FLAMEDEFS+`<rect x="8" y="20" width="120" height="40" rx="8" class="box-pyo"/><text x="68" y="45" text-anchor="middle" class="lbl">Pyomo: F, J</text><rect x="252" y="20" width="120" height="40" rx="8" class="box-jax"/><text x="312" y="45" text-anchor="middle" class="lbl">JAX: F, J</text>
  <path d="M68,60 C68,96 140,92 156,110" class="ln-pyo" marker-end="url(#ahp)"/><path d="M312,60 C312,96 240,92 222,110" class="ln-jax" marker-end="url(#ahj)"/>`+T(112,160,brick(0,0,0,5,2,34,"var(--c-engine)",{label:"Newton + MUMPS",fs:12}))+nozzle(150,160,30,14)+nozzle(210,160,30,14)+flame(150,174,52,14)+flame(210,174,52,14)),
 flight:()=>svg("0 0 300 220",flight(10,30,170)+pilot(240,200,1.5)),
 time_bar:()=>svg("0 0 960 150",`<text x="20" y="24" class="lbl" font-weight="700">One ALTERTAX run on 20x41, about 1.6 to 1.8 min after the cuts (2026-11-12)</text>
  <rect x="20" y="40" width="920" height="40" rx="6" class="box"/><rect x="20" y="40" width="166" height="40" rx="6" class="box-jax"/><text x="103" y="65" text-anchor="middle" class="lbl">JAX + MUMPS</text>
  <text x="560" y="65" text-anchor="middle" class="lbl">Python: prepare, pair, load, write and compare results</text>
  <text x="20" y="104" class="sm">≈ 0.25 to 0.33 min</text><text x="940" y="104" text-anchor="end" class="sm">≈ 1.3 to 1.5 min</text>
  <text x="20" y="132" class="sm">The math is the small part. The cuts of these weeks went after the big one.</text>`,"wide"),
 storage_box:()=>svg("0 0 340 260",storageBox(30,60,230,180,["&lt;key&gt;/records.pkl","&lt;key&gt;/state.pkl","&lt;key&gt;/pairing-*.npy","jax_aot/*.aot","jax/","photos/"])+`<text x="30" y="30" class="mono sm">~/.cache/equilibria/family</text>`),
 box_photo:()=>svg("0 0 380 260",FLAMEDEFS+boxPhoto(16,14,150,230)+T(250,236,`<g transform="scale(.62)">${rocket(0,0,0,true).replace(/^<g transform="translate\(0,0\)">/,'<g>')}</g>`)+`<text x="208" y="128" text-anchor="middle" class="disp" style="font-size:26px;font-weight:600;fill:var(--jxb-ok)">=</text><text x="208" y="150" text-anchor="middle" class="mono sm">≤ 1e-9</text>`)
};

const TITLES = {
  stack: "GTAP blocks stacked as Lego bricks on the data plate, closed by the closure tile",
  parts: "The pieces: brick, plate, smooth tile, instruction sheet, flight plan, pilot, rocket, and the JAX pieces clay, mold, storage box, box photo and engine",
  big_picture: "The blocks feed two roads, Pyomo and JAX, that end in the same Newton engine with MUMPS",
  same_brick: "One brick, eq_xaa_activity, can go to a Pyomo model or to a RecordingModel",
  two_objects: "The same rule receives a Pyomo model or a RecordingModel and returns a constraint or a tree",
  press: "A brick pressed into clay leaves a mold: one tree per cell",
  tree: "The tree recorded for one cell of eq_xaa_activity",
  one_mold: "One mold poured with vmap gives many equal pieces",
  families: "Cells with the same shape become one mold plus a table, and JAX computes F and J for every row at once",
  engine: "Pyomo and JAX both hand F and J to the same Newton plus MUMPS engine",
  flight: "The flight plan, base, check and shock, flown by the pilot",
  time_bar: "In one ALTERTAX run on 20x41, JAX and MUMPS take the small part of the time",
  storage_box: "The storage box on disk and its drawers",
  box_photo: "The built rocket compared with the photo on its box, within 1e-9",
};

const out = process.argv[2];
if (!out) {
  console.error("usage: export_jax_doc_svgs.mjs <output dir>");
  process.exit(2);
}
mkdirSync(out, { recursive: true });
for (const [name, fig] of Object.entries(FIGS)) {
  const file = join(out, `${name.replaceAll("_", "-")}.svg`);
  const slug = name.replaceAll("_", "-");
  const body = fig()
    .replace(/\bid="([^"]+)"/g, `id="${slug}-$1"`)
    .replace(/url\(#([^)]+)\)/g, `url(#${slug}-$1)`)
    .replace(/^(<svg [^>]*>)/, `$1<title>${TITLES[name]}</title>`);
  writeFileSync(file, body + "\n");
  console.log(file);
}
