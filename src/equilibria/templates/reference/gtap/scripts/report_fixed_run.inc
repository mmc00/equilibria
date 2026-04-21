* -------------------------------------------------------------------------
*  Report fixed/free counts by variable - run for a given time slice.
*  Usage: $$batinclude "report_fixed_run.inc" check
*  
*  NOTE: This file does NOT call mvar.gms to avoid macro redefinition
*  issues with Action=C compilation. Instead, it directly generates
*  report statements for key variables.
* -------------------------------------------------------------------------

$setargs reportYear

* Count fixed/free for key model variables
* Regional variables (r, ...)

total_count("nd", "%reportYear%") = sum((r,a), 1);
fixed_count("nd", "%reportYear%") = sum((r,a)$(abs(nd.lo(r,a,"%reportYear%") - nd.up(r,a,"%reportYear%")) <= 1e-12), 1);
free_count("nd", "%reportYear%") = total_count("nd", "%reportYear%") - fixed_count("nd", "%reportYear%");

total_count("va", "%reportYear%") = sum((r,a), 1);
fixed_count("va", "%reportYear%") = sum((r,a)$(abs(va.lo(r,a,"%reportYear%") - va.up(r,a,"%reportYear%")) <= 1e-12), 1);
free_count("va", "%reportYear%") = total_count("va", "%reportYear%") - fixed_count("va", "%reportYear%");

total_count("px", "%reportYear%") = sum((r,a), 1);
fixed_count("px", "%reportYear%") = sum((r,a)$(abs(px.lo(r,a,"%reportYear%") - px.up(r,a,"%reportYear%")) <= 1e-12), 1);
free_count("px", "%reportYear%") = total_count("px", "%reportYear%") - fixed_count("px", "%reportYear%");

total_count("xa", "%reportYear%") = sum((r,i,aa), 1);
fixed_count("xa", "%reportYear%") = sum((r,i,aa)$(abs(xa.lo(r,i,aa,"%reportYear%") - xa.up(r,i,aa,"%reportYear%")) <= 1e-12), 1);
free_count("xa", "%reportYear%") = total_count("xa", "%reportYear%") - fixed_count("xa", "%reportYear%");

total_count("xf", "%reportYear%") = sum((r,fp,a), 1);
fixed_count("xf", "%reportYear%") = sum((r,fp,a)$(abs(xf.lo(r,fp,a,"%reportYear%") - xf.up(r,fp,a,"%reportYear%")) <= 1e-12), 1);
free_count("xf", "%reportYear%") = total_count("xf", "%reportYear%") - fixed_count("xf", "%reportYear%");

total_count("x", "%reportYear%") = sum((r,a,i), 1);
fixed_count("x", "%reportYear%") = sum((r,a,i)$(abs(x.lo(r,a,i,"%reportYear%") - x.up(r,a,i,"%reportYear%")) <= 1e-12), 1);
free_count("x", "%reportYear%") = total_count("x", "%reportYear%") - fixed_count("x", "%reportYear%");

total_count("xp", "%reportYear%") = sum((r,a), 1);
fixed_count("xp", "%reportYear%") = sum((r,a)$(abs(xp.lo(r,a,"%reportYear%") - xp.up(r,a,"%reportYear%")) <= 1e-12), 1);
free_count("xp", "%reportYear%") = total_count("xp", "%reportYear%") - fixed_count("xp", "%reportYear%");

total_count("pp", "%reportYear%") = sum((r,a,i), 1);
fixed_count("pp", "%reportYear%") = sum((r,a,i)$(abs(pp.lo(r,a,i,"%reportYear%") - pp.up(r,a,i,"%reportYear%")) <= 1e-12), 1);
free_count("pp", "%reportYear%") = total_count("pp", "%reportYear%") - fixed_count("pp", "%reportYear%");

total_count("ps", "%reportYear%") = sum((r,i), 1);
fixed_count("ps", "%reportYear%") = sum((r,i)$(abs(ps.lo(r,i,"%reportYear%") - ps.up(r,i,"%reportYear%")) <= 1e-12), 1);
free_count("ps", "%reportYear%") = total_count("ps", "%reportYear%") - fixed_count("ps", "%reportYear%");

total_count("ytax", "%reportYear%") = sum((r,gy), 1);
fixed_count("ytax", "%reportYear%") = sum((r,gy)$(abs(ytax.lo(r,gy,"%reportYear%") - ytax.up(r,gy,"%reportYear%")) <= 1e-12), 1);
free_count("ytax", "%reportYear%") = total_count("ytax", "%reportYear%") - fixed_count("ytax", "%reportYear%");

total_count("yc", "%reportYear%") = sum(r, 1);
fixed_count("yc", "%reportYear%") = sum(r$(abs(yc.lo(r,"%reportYear%") - yc.up(r,"%reportYear%")) <= 1e-12), 1);
free_count("yc", "%reportYear%") = total_count("yc", "%reportYear%") - fixed_count("yc", "%reportYear%");

total_count("u", "%reportYear%") = sum(r, 1);
fixed_count("u", "%reportYear%") = sum(r$(abs(u.lo(r,"%reportYear%") - u.up(r,"%reportYear%")) <= 1e-12), 1);
free_count("u", "%reportYear%") = total_count("u", "%reportYear%") - fixed_count("u", "%reportYear%");

total_count("rsav", "%reportYear%") = sum(r, 1);
fixed_count("rsav", "%reportYear%") = sum(r$(abs(rsav.lo(r,"%reportYear%") - rsav.up(r,"%reportYear%")) <= 1e-12), 1);
free_count("rsav", "%reportYear%") = total_count("rsav", "%reportYear%") - fixed_count("rsav", "%reportYear%");

* Trade variables
total_count("xw", "%reportYear%") = sum((r,i,rp), 1);
fixed_count("xw", "%reportYear%") = sum((r,i,rp)$(abs(xw.lo(r,i,rp,"%reportYear%") - xw.up(r,i,rp,"%reportYear%")) <= 1e-12), 1);
free_count("xw", "%reportYear%") = total_count("xw", "%reportYear%") - fixed_count("xw", "%reportYear%");

total_count("pe", "%reportYear%") = sum((r,i,rp), 1);
fixed_count("pe", "%reportYear%") = sum((r,i,rp)$(abs(pe.lo(r,i,rp,"%reportYear%") - pe.up(r,i,rp,"%reportYear%")) <= 1e-12), 1);
free_count("pe", "%reportYear%") = total_count("pe", "%reportYear%") - fixed_count("pe", "%reportYear%");

total_count("pm", "%reportYear%") = sum((r,i,rp), 1);
fixed_count("pm", "%reportYear%") = sum((r,i,rp)$(abs(pm.lo(r,i,rp,"%reportYear%") - pm.up(r,i,rp,"%reportYear%")) <= 1e-12), 1);
free_count("pm", "%reportYear%") = total_count("pm", "%reportYear%") - fixed_count("pm", "%reportYear%");

total_count("imptx", "%reportYear%") = sum((r,i,rp), 1);
fixed_count("imptx", "%reportYear%") = sum((r,i,rp)$(abs(imptx.lo(r,i,rp,"%reportYear%") - imptx.up(r,i,rp,"%reportYear%")) <= 1e-12), 1);
free_count("imptx", "%reportYear%") = total_count("imptx", "%reportYear%") - fixed_count("imptx", "%reportYear%");

total_count("exptx", "%reportYear%") = sum((r,i,rp), 1);
fixed_count("exptx", "%reportYear%") = sum((r,i,rp)$(abs(exptx.lo(r,i,rp,"%reportYear%") - exptx.up(r,i,rp,"%reportYear%")) <= 1e-12), 1);
free_count("exptx", "%reportYear%") = total_count("exptx", "%reportYear%") - fixed_count("exptx", "%reportYear%");

* Factor markets
total_count("pf", "%reportYear%") = sum((r,fp,a), 1);
fixed_count("pf", "%reportYear%") = sum((r,fp,a)$(abs(pf.lo(r,fp,a,"%reportYear%") - pf.up(r,fp,a,"%reportYear%")) <= 1e-12), 1);
free_count("pf", "%reportYear%") = total_count("pf", "%reportYear%") - fixed_count("pf", "%reportYear%");

total_count("pft", "%reportYear%") = sum((r,fm), 1);
fixed_count("pft", "%reportYear%") = sum((r,fm)$(abs(pft.lo(r,fm,"%reportYear%") - pft.up(r,fm,"%reportYear%")) <= 1e-12), 1);
free_count("pft", "%reportYear%") = total_count("pft", "%reportYear%") - fixed_count("pft", "%reportYear%");

* Global variables
total_count("rorg", "%reportYear%") = 1;
fixed_count("rorg", "%reportYear%") = (abs(rorg.lo("%reportYear%") - rorg.up("%reportYear%")) <= 1e-12);
free_count("rorg", "%reportYear%") = total_count("rorg", "%reportYear%") - fixed_count("rorg", "%reportYear%");

total_count("pnum", "%reportYear%") = 1;
fixed_count("pnum", "%reportYear%") = (abs(pnum.lo("%reportYear%") - pnum.up("%reportYear%")) <= 1e-12);
free_count("pnum", "%reportYear%") = total_count("pnum", "%reportYear%") - fixed_count("pnum", "%reportYear%");

* Price numeraire related
total_count("pabs", "%reportYear%") = sum(r, 1);
fixed_count("pabs", "%reportYear%") = sum(r$(abs(pabs.lo(r,"%reportYear%") - pabs.up(r,"%reportYear%")) <= 1e-12), 1);
free_count("pabs", "%reportYear%") = total_count("pabs", "%reportYear%") - fixed_count("pabs", "%reportYear%");

execute_unload "FIX_FREE_%reportYear%.gdx", fixed_count, free_count, total_count;
