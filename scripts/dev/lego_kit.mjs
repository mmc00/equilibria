// The Lego drawing kit shared by the guide figures (export_jax_doc_svgs.mjs,
// export_architecture_svgs.mjs). From the equilibria Lego mockup (2026-11-12):
// oblique projection with U=26 (stud pitch), H=24 (brick height), K=0.36 (depth).
// Every colour is a CSS variable or a class of docs/site/_static/jax_backend.css,
// so the inlined SVGs follow the site's light/dark toggle. One concept is always
// the same piece and colour.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export const U=26, H=24, K=0.36; // stud pitch, brick height, oblique depth factor
export const P=(x,y,z)=>[x+K*y, -z-K*y];
export const pt=a=>a.map(v=>v.toFixed(1)).join(",");
export const poly=(pts,fill)=>`<polygon points="${pts.map(pt).join(" ")}" style="fill:${fill}"/>`;
export const top=c=>`color-mix(in oklab, ${c} 74%, white)`;
export const side=c=>`color-mix(in oklab, ${c} 70%, black)`;
export const studSide=c=>`color-mix(in oklab, ${c} 82%, black)`;
export const T=(x,y,b)=>`<g transform="translate(${x},${y})">${b}</g>`;
export const svg=(vb,body,cls="")=>`<svg viewBox="${vb}" role="img" xmlns="http://www.w3.org/2000/svg" class="jxb ${cls}">${body}</svg>`;
export const ARR=`<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="head"/></marker><marker id="ahj" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="head-jax"/></marker><marker id="ahp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="head-pyo"/></marker></defs>`;
export const FLAMEDEFS=`<defs><linearGradient id="fl-out" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:var(--flame-3)"/><stop offset=".55" style="stop-color:var(--flame-4)"/><stop offset="1" style="stop-color:var(--flame-5);stop-opacity:0"/></linearGradient><linearGradient id="fl-in" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:var(--flame-1)"/><stop offset=".6" style="stop-color:var(--flame-2)"/><stop offset="1" style="stop-color:var(--flame-3);stop-opacity:0"/></linearGradient><radialGradient id="smoke" cx=".5" cy=".5" r=".5"><stop offset="0" style="stop-color:var(--smoke);stop-opacity:.9"/><stop offset="1" style="stop-color:var(--smoke);stop-opacity:0"/></radialGradient></defs>`;

export function studs(x,y,z,nx,ny,c){let s="";const r=U*.3,sh=U*.2;
  for(let j=ny-1;j>=0;j--)for(let i=0;i<nx;i++){const [cx,cy]=P(x+(i+.5)*U,y+(j+.5)*U,z);
    s+=`<path d="M${cx-r},${cy} L${cx-r},${cy-sh} A${r},${r*.42} 0 0 1 ${cx+r},${cy-sh} L${cx+r},${cy} A${r},${r*.42} 0 0 1 ${cx-r},${cy}Z" style="fill:${studSide(c)}"/>`;
    s+=`<ellipse cx="${cx}" cy="${cy-sh}" rx="${r}" ry="${r*.42}" style="fill:${top(c)}"/>`;}
  return s;}
export function brick(x,y,z,nx,ny,h,c,o={}){const w=nx*U,d=ny*U;let s="";
  s+=poly([P(x,y,z+h),P(x+w,y,z+h),P(x+w,y+d,z+h),P(x,y+d,z+h)],top(c));
  s+=poly([P(x+w,y,z),P(x+w,y+d,z),P(x+w,y+d,z+h),P(x+w,y,z+h)],side(c));
  s+=poly([P(x,y,z),P(x+w,y,z),P(x+w,y,z+h),P(x,y,z+h)],c);
  if(o.studs!==false)s+=studs(x,y,z+h,nx,ny,c);
  if(o.label){const [lx,ly]=P(x+w/2,y,z+h/2);s+=`<text x="${lx}" y="${ly+4.5}" text-anchor="middle" class="on-brick" style="font-size:${o.fs||12.5}px">${o.label}</text>`;}
  return s;}
export const plate=(x,y,z,nx,ny,c)=>brick(x,y,z,nx,ny,8,c);
export function tile(x,y,z,nx,ny,c,o={}){let s=brick(x,y,z,nx,ny,o.h||14,c,{studs:false,label:o.label,fs:o.fs});
  const [a1,a2]=P(x,y,z+2),[b1,b2]=P(x+nx*U,y,z+2);
  s+=`<line x1="${a1}" y1="${a2}" x2="${b1}" y2="${b2}" style="stroke:color-mix(in oklab, ${c} 60%, black);stroke-width:1"/>`;return s;}
export function sheet(x,y,w,h,rows){
  let o=`<path d="M${x},${y} H${x+w-16} L${x+w},${y+16} V${y+h} H${x}Z" style="fill:var(--jxb-chip);stroke:var(--jxb-line);stroke-width:1.2"/><path d="M${x+w-16},${y} V${y+16} H${x+w}" style="fill:var(--jxb-line)"/>`;
  o+=`<text x="${x+12}" y="${y+22}" class="disp" style="font-size:13px;font-weight:600">GTAP</text>`;
  rows.forEach((r,i)=>{const yy=y+34+i*15;o+=`<rect x="${x+12}" y="${yy}" width="16" height="9" rx="2" style="fill:${r[0]}"/><rect x="${x+34}" y="${yy+3}" width="${r[1]}" height="3" rx="1.5" style="fill:var(--jxb-muted);opacity:.55"/>`;});
  return o;}
export function flight(x,y,w){const st=[["base","not solved"],["check","solved"],["shock","solved"]];
  let o=`<rect x="${x}" y="${y}" width="${w}" height="150" rx="10" style="fill:var(--jxb-chip);stroke:var(--jxb-line);stroke-width:1.2"/><path d="M${x+28},${y+34} V${y+124}" style="stroke:var(--pyo);stroke-width:2.5"/>`;
  st.forEach((s,i)=>{const yy=y+34+i*45;
    o+=`<circle cx="${x+28}" cy="${yy}" r="8" style="fill:${i==2?'var(--c-shock)':'var(--pyo)'}"/>`;
    o+=`<text x="${x+44}" y="${yy+1}" class="disp" style="font-size:14px;font-weight:600">${s[0]}</text><text x="${x+44}" y="${yy+15}" class="mono sm">${s[1]}</text>`;});
  return o;}
export function pilot(cx,by,s=1){const q=v=>v*s;let o="";
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
export function flame(cx,cy,L,w){return `<path d="M${cx-w},${cy} C${cx-w},${cy+L*.35} ${cx-w*.35},${cy+L*.7} ${cx},${cy+L} C${cx+w*.35},${cy+L*.7} ${cx+w},${cy+L*.35} ${cx+w},${cy}Z" fill="url(#fl-out)"/><path d="M${cx-w*.55},${cy} C${cx-w*.55},${cy+L*.3} ${cx-w*.2},${cy+L*.5} ${cx},${cy+L*.62} C${cx+w*.2},${cy+L*.5} ${cx+w*.55},${cy+L*.3} ${cx+w*.55},${cy}Z" fill="url(#fl-in)"/>`;}
export function nozzle(cx,cy,w,h){return `<path d="M${cx-w*.35},${cy} L${cx+w*.35},${cy} L${cx+w*.6},${cy+h} L${cx-w*.6},${cy+h}Z" style="fill:var(--c-engine)"/><path d="M${cx-w*.35},${cy} L${cx-w*.05},${cy} L${cx-w*.2},${cy+h} L${cx-w*.6},${cy+h}Z" style="fill:var(--c-engine-lt)"/><ellipse cx="${cx}" cy="${cy+h}" rx="${w*.6}" ry="${w*.16}" style="fill:var(--c-engine-dk)"/>`;}
export const FAM=[["Shock","--c-shock"],["Export (CET)","--c-trade"],["Production","--c-prod"],["Factors","--c-factor"],["Import","--c-trade"],["Households","--c-hh"],["Income, gov.","--c-gov"],["Closure","--c-closure"]];
export function rocket(cx,base,fl=1,small=false){const nx=2,w=nx*U,hh=small?14:18;let o="";const x=cx-w/2-K*U/2;
  o+=plate(x-U*1.5,0,0,nx+3,2,"var(--c-data)");let z=8;const lift=fl?70:0;
  if(fl){const [fx,fy]=P(x+w/2,U,z+lift);o+=`<ellipse cx="${fx-30}" cy="${fy+lift-6}" rx="26" ry="12" fill="url(#smoke)"/><ellipse cx="${fx+34}" cy="${fy+lift-4}" rx="30" ry="13" fill="url(#smoke)"/>`+nozzle(fx,fy,34,12)+flame(fx,fy+12,lift-6,16);}
  z+=lift;const [f1x,f1y]=P(x,U,z),[f2x,f2y]=P(x+w,U,z);
  o+=`<path d="M${f1x},${f1y-30} L${f1x-22},${f1y+4} L${f1x},${f1y-4}Z" style="fill:${side('var(--c-shock)')}"/>`;
  FAM.forEach((f,i)=>{o+=brick(x,0,z,nx,2,hh,`var(${f[1]})`,{studs:i==FAM.length-1});z+=hh;});
  o+=`<path d="M${f2x},${f2y-30} L${f2x+22},${f2y+4} L${f2x},${f2y-4}Z" style="fill:var(--c-shock)"/>`;
  const a=P(x,0,z),b=P(x+w,0,z),c=P(x+w,2*U,z),tip=P(x+w/2,U,z+46);
  o+=poly([a,b,tip],"var(--c-closure)")+poly([b,c,tip],side("var(--c-closure)"));
  return T(0,base,o);}


// Write each figure as <out>/<name with - for _>.svg: ids prefixed with the slug
// (the figures are inlined into one page) and the title as the text alternative.
export function writeFigures(out, figs, titles) {
  mkdirSync(out, { recursive: true });
  for (const [name, fig] of Object.entries(figs)) {
    const slug = name.replaceAll("_", "-");
    const file = join(out, `${slug}.svg`);
    const body = fig()
      .replace(/\bid="([^"]+)"/g, `id="${slug}-$1"`)
      .replace(/url\(#([^)]+)\)/g, `url(#${slug}-$1)`)
      .replace(/^(<svg [^>]*>)/, `$1<title>${titles[name]}</title>`);
    writeFileSync(file, body + "\n");
    console.log(file);
  }
}
