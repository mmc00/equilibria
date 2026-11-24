#!/usr/bin/env node
// Writes the figures of docs/site/guide/architecture.md as SVG files.
//
//   node scripts/dev/export_architecture_svgs.mjs docs/assets/architecture
//
// Same Lego kit (lego_kit.mjs) and colours (jax_backend.css) as the JAX page, so a
// piece means the same thing on both pages. Each title says everything the
// drawing shows: an LLM reading the page does not see the image.
import { U, K, P, T, svg, ARR, FLAMEDEFS, brick, plate, tile, sheet, flight, pilot, rocket, tower, TOWER, writeFigures } from "./lego_kit.mjs";

const arrow=(x1,y1,x2,y2)=>`<path d="M${x1},${y1} L${x2},${y2}" class="ln" marker-end="url(#ah)"/>`;
const txt=(x,y,t,cls="lbl",a="start")=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${a}">${t}</text>`;
const chip=(x,y,w,h,title,sub)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" class="box"/>`+txt(x+w/2,y+22,title,"disp lbl","middle")+(sub?txt(x+w/2,y+40,sub,"mono sm","middle"):"");
// a small tower: nx=4 studs, no text on the bricks
const mini=(o={})=>tower({nx:4,h:15,labels:false,...o});

const FIGS={
 tower:()=>svg("0 20 560 245",T(40,250,tower({nx:7,h:26,fs:12.5,codeNames:true}))),

 periods:()=>{let o=ARR;
  o+=T(40,215,mini())+txt(92,250,"one period","disp lbl","middle")+txt(92,266,"the SP model","mono sm","middle");
  o+=arrow(190,140,262,140)+txt(226,128,"copy x3","sm","middle");
  [["base","not solved"],["check","solved"],["shock","solved"]].forEach((p,i)=>{const x=280+i*180;
    o+=T(x,215,mini())+txt(x+52,250,p[0],"disp lbl","middle")+txt(x+52,266,`t = '${p[0]}'`,"mono sm","middle");});
  // the shock: one instrument cell of the shock copy differs from check
  const [sx,sy]=P(4*U,2*U,8+15/2);
  o+=`<circle cx="${640+sx+12}" cy="${215+sy}" r="7" style="fill:var(--c-shock);stroke:var(--jxb-surface);stroke-width:2"/>`;
  o+=txt(640+sx+26,215+sy+4,"imptx +10 %","mono sm");
  o+=txt(640+sx+26,215+sy+20,"the one cell","sm")+txt(640+sx+26,215+sy+34,"that differs","sm");
  return svg("0 55 900 220",o);},

 overwrite:()=>{let o=ARR;const nx=5,h=19,fs=10.5;
  o+=T(40,240,tower({nx,h,fs}))+txt(110,268,"check","disp lbl","middle");
  o+=T(330,240,tower({nx,h,fs,top:{label:"Closure + eq_wreal",colour:"var(--c-gen)"}}))+txt(400,268,"shock","disp lbl","middle");
  // the old tile, lifted off the shock tower
  o+=T(560,95,tile(0,0,0,nx,2,"var(--c-closure)",{h:h*.7,label:"Closure",fs}));
  o+=`<path d="M590,108 C560,108 520,60 470,62" class="ln dash" marker-end="url(#ah)"/>`;
  o+=txt(560,40,"@overwrite(ClosureBlock,","mono sm")+txt(560,55,'           period="shock")',"mono sm");
  o+=txt(560,150,"one block, one period:","sm")+txt(560,165,"base and check keep","sm")+txt(560,180,"the original","sm");
  return svg("0 20 760 260",o);},

 recipe:()=>{let o=ARR;
  o+=txt(20,24,"gtap_static(...)","mono sm")+flight(20,34,170);
  o+=pilot(260,200,1.1)+txt(260,222,"the driver","disp lbl","middle")+txt(260,238,"core/driver.py","mono sm","middle");
  o+=arrow(300,112,350,112);
  const st=[["before","freeze, seed,","recalibrate, fix"],["solve","once, in steps","or Johansen"],["after","post-solve,","report"]];
  st.forEach((s,i)=>{const x=360+i*140;o+=`<rect x="${x}" y="70" width="120" height="84" rx="10" class="box"/>`+txt(x+60,94,s[0],"disp lbl","middle")+txt(x+60,116,s[1],"sm","middle")+txt(x+60,132,s[2],"sm","middle");
    if(i<2)o+=arrow(x+122,112,x+138,112);});
  o+=txt(570,186,"for each period: base, then check, then shock","sm","middle");
  return svg("0 0 800 250",o);},

 pipeline:()=>{let o=ARR+FLAMEDEFS;const y=150,cx=i=>70+i*190;
  const cap=(i,a,b)=>txt(cx(i),y+40,a,"disp lbl","middle")+txt(cx(i),y+57,b,"mono sm","middle");
  o+=T(cx(0)-64,y,plate(0,0,0,4,2,"var(--c-data)"))+cap(0,"Data","SAM, HAR, GDX, MIP");
  o+=T(cx(1)-60,y,plate(0,0,0,4,2,"var(--c-data)")+brick(U,0,8,2,2,20,"var(--c-prod)"))+cap(1,"Calibration","benchmark → params");
  o+=T(cx(2)-50,y,tower({nx:3,h:11,labels:false}))+cap(2,"Blocks → model","gtap, pep, simple_open");
  o+=pilot(cx(3),y+4,.9)+cap(3,"Driver + solver","PATH, Newton, IPOPT");
  o+=`<rect x="${cx(4)-46}" y="${y-104}" width="92" height="100" rx="6" style="fill:var(--c-closure)"/><rect x="${cx(4)-38}" y="${y-80}" width="76" height="68" rx="4" style="fill:var(--c-photo)"/>`+`<text x="${cx(4)}" y="${y-87}" class="on-brick" text-anchor="middle" style="font-size:10px">GAMS, GEMPACK</text>`;
  o+=`<g transform="translate(${cx(4)-28},${y-16}) scale(.6)">${tower({nx:2,h:8,labels:false})}</g>`;
  o+=cap(4,"Validation","gates vs the reference");
  for(let i=0;i<4;i++)o+=arrow(cx(i)+70,y-40,cx(i+1)-62,y-40);
  return svg("0 30 960 195",o);},
};

const TITLES={
 tower:"GTAP as a tower of eight Lego bricks on the data plate, the blocks of GTAP_BLOCK_ORDER from bottom to top: ShockBlock (the instruments), TradeCETBlock (export, CET), ProductionSupplyBlock (production), FactorBlock (factor markets), ArmingtonBilateralBlock (imports by origin), DemandUtilityBlock (households), IncomeBlock (income and government), and ClosureBlock as the smooth tile on top that closes the model",
 periods:"One single-period model, the tower, is copied three times into the periods base, check and shock; the copies are identical except one instrument cell of the shock copy, for example imptx +10 %, which is the shock",
 overwrite:"The check and shock towers are equal except the top: in the shock period @overwrite(ClosureBlock, period=\"shock\") lifts off the closure tile and puts a closure with the extra equation eq_wreal; base and check keep the original",
 recipe:"The flight plan gtap_static lists the periods base (not solved), check and shock (solved); the pilot, the driver in core/driver.py, flies each period in order: steps before the solve (freeze, seed, recalibrate, fix), the solve (once, in steps or Johansen), and steps after (post-solve, report)",
 pipeline:"The pipeline in five stations: data (SAM, HAR, GDX, MIP) on a plate, calibration fits the parameters to the benchmark, the blocks build the model tower, the driver and a solver (PATH, Newton, IPOPT) fly it, and validation compares the result with the GAMS and GEMPACK reference",
};

const out = process.argv[2];
if (!out) {
  console.error("usage: export_architecture_svgs.mjs <output dir>");
  process.exit(2);
}
writeFigures(out, FIGS, TITLES);
