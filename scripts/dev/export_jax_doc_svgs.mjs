#!/usr/bin/env node
// Writes the figures of docs/site/guide/jax_backend.md as SVG files.
//
//   node scripts/dev/export_jax_doc_svgs.mjs docs/assets/jax-backend
//
// The drawing kit (lego_kit.mjs) comes from the equilibria Lego mockup (2026-11-12):
// oblique projection with U=26 (stud pitch), H=24 (brick height), K=0.36 (depth). Every
// colour is a CSS variable or a class from docs/site/_static/jax_backend.css, so
// the inlined SVGs follow the site's light/dark toggle. One concept is always the
// same piece and colour: see the "parts" figure.
import { U, H, K, P, poly, top, side, T, svg, ARR, FLAMEDEFS, brick, plate, tile, sheet, flight, pilot, flame, nozzle, rocket, tower, writeFigures } from "./lego_kit.mjs";

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
 stack:()=>svg("0 0 290 260",T(28,240,tower({nx:7,h:24,fs:11.5}))),
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
  stack: "GTAP as eight Lego bricks on the data plate, bottom to top: shock, export (CET), production, factors, imports, households, income and government, and the closure tile on top",
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
writeFigures(out, FIGS, TITLES);
