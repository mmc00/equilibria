* --------------------------------------------------------------------------------------------------
*
*     Read in the GTAP SETS
*
* --------------------------------------------------------------------------------------------------

sets
   acts           "Activities"
   comm           "Commodities"
   marg(comm)     "Margin commodities"
   reg            "Regions"
   endw           "Endowments"
   endwf(endw)    "Fixed factors"
   endwm(endw)    "Mobile factors"
   endws(endw)    "Sluggish factors"
;

$gdxin "%inDir%/%BaseName%Sets.gdx"
$load acts, comm, marg, reg, endw, endwf, endwm, endws
$gdxin

*  CREATE THE SAM SETS

set stdlab "Standard SAM labels" /
   TRD               "Trade account"
   hhd               "Household"
   gov               "Government"
   inv               "Investment"
   deprY             "Depreciation"
   tmg               "Trade margins"
   itax              "Indirect tax"
   ptax              "Production tax"
   mtax              "Import tax"
   etax              "Export tax"
   vtax              "Taxes on factors of production"
   vsub              "Subsidies on factors of production"
   dtax              "Direct taxation"
   ctax              "Carbon tax"
   bop               "Balance of payments account"
   tot               "Total for row/column sums"
/ ;

set findem(stdlab) "Final demand accounts" /
   hhd               "Household"
   gov               "Government"
   inv               "Investment"
   tmg               "Trade margins"
/ ;

set is "SAM accounts for aggregated SAM" /

*  User-defined activities

   set.acts

*  User-defined commodities

   set.comm

*  User-defined factors

   set.endw

*  Standard SAM accounts

   set.stdlab

*  User-defined regions

   set.reg

/ ;

alias(is, js) ;

set aa(is) "Armington agents" /

   set.acts

   set.findem

/ ;

set a(aa) "Activities" /

   set.acts

/ ;

set i(is) "Commodities" /

   set.comm

/ ;
alias(i, j) ;

set r(is) "Regions" /

   set.reg

/ ;

alias(r,s) ; alias(r,d) ; alias(r,rp) ;

set fp(is)  "Factors of production" /

   set.endw

/ ;

set fnm(fp) "Non-mobile factors" ;
loop((fp,endwf)$sameas(fp,endwf),
   fnm(fp) = yes ;
) ;

set fm(fp) "Mobile factors" ;
fm(fp)$(not fnm(fp)) = yes ;


set fd(aa) "Domestic final demand agents" /

   set.findem

/ ;

set h(fd) "Households" /
   hhd               "Household"
/ ;

set gov(fd) "Government" /
   gov               "Government"
/ ;

set inv(fd) "Investment" /
   inv               "Investment"
/ ;

set fdc(fd) "Final demand accounts with CES expenditure function" /
   gov               "Government"
   inv               "Investment"
/ ;

set tmg(fd) "Domestic supply of trade margins services" /
   tmg               "Trade margins"
/ ;

alias(i0,i) ; alias(a0,a) ; alias(i,j) ; alias(j0,i0) ;

sets
   mapa0(a,a0)
   mapi0(i,i0)
;

*  No aggregation needed for this version of the model--so map a0 to a and i0 to i

mapa0(a,a) = yes ;
mapi0(i,i) = yes ;

* --------------------------------------------------------------------------------------------------
*
*     Read in the GTAP database
*
* --------------------------------------------------------------------------------------------------

parameters
   VDFB(i0, a0, r)      "Firm purchases of domestic goods at basic prices"
   VDFP(i0, a0, r)      "Firm purchases of domestic goods at purchaser prices"
   VMFB(i0, a0, r)      "Firm purchases of imported goods at basic prices"
   VMFP(i0, a0, r)      "Firm purchases of domestic goods at purchaser prices"
   VDPB(i0, r)          "Private purchases of domestic goods at basic prices"
   VDPP(i0, r)          "Private purchases of domestic goods at purchaser prices"
   VMPB(i0, r)          "Private purchases of imported goods at basic prices"
   VMPP(i0, r)          "Private purchases of domestic goods at purchaser prices"
   VDGB(i0, r)          "Government purchases of domestic goods at basic prices"
   VDGP(i0, r)          "Government purchases of domestic goods at purchaser prices"
   VMGB(i0, r)          "Government purchases of imported goods at basic prices"
   VMGP(i0, r)          "Government purchases of domestic goods at purchaser prices"
   VDIB(i0, r)          "Investment purchases of domestic goods at basic prices"
   VDIP(i0, r)          "Investment purchases of domestic goods at purchaser prices"
   VMIB(i0, r)          "Investment purchases of imported goods at basic prices"
   VMIP(i0, r)          "Investment purchases of domestic goods at purchaser prices"

   EVFB(fp, a0, r)      "Primary factor purchases at basic prices"
   EVFP(fp, a0, r)      "Primary factor purchases at purchaser prices"
   EVOS(fp, a0, r)      "Factor remuneration after income tax"

   VXSB(i0, r, rp)      "Exports at basic prices"
   VFOB(i0, r, rp)      "Exports at FOB prices"
   VCIF(i0, r, rp)      "Import at CIF prices"
   VMSB(i0, r, rp)      "Imports at basic prices"

   VST(i0, r)           "Exports of trade and transport services"
   VTWR(i0, j0, r, rp)  "Margins by margin commodity"

   SAVE(r)              "Net saving, by region"
   VDEP(r)              "Capital depreciation"
   VKB(r)               "Capital stock"
   POP0(r)              "Population"

   MAKS(i0,a0,r)        "Make matrix at supply prices"
   MAKB(i0,a0,r)        "Make matrix at basic prices (incl taxes)"
   PTAX(i0,a0,r)        "Output taxes"

   fbep(fp, a0, r)      "Factor subsidies"
   ftrv(fp, a0, r)      "Tax on factor use"
   tvom(a0,r)           "Value of output"
   check(a0,r)          "Check"
;

* Load the GTAP data at compile time so NEOS remote execution does not need
* access to the local GDX files during the execution phase.
$gdxin "%inDir%/%BaseName%Dat.gdx"
$load vdfb vdfp vmfb vmfp
$load vdpb vdpp vmpb vmpp
$load vdgb vdgp vmgb vmgp
$load vdib vdip vmib vmip
$load evfb evfp evos
$load vxsb vfob vcif vmsb
$load vst vtwr
$load save vdep vkb pop0=pop
$load maks makb
$gdxin

*  !!!! MAY WANT TO FIX THIS AT SOME STAGE--THERE IS INCONSISTENCY IN THE
*        HANDLING OF FBEP and FTRV

fbep(fp,a0,r) = 0 ;
ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;
ptax(i0,a0,r) = makb(i0,a0,r) - maks(i0,a0,r) ;

* --------------------------------------------------------------------------------------------------
*
*     Read in CO2 emissions data
*
* --------------------------------------------------------------------------------------------------

Parameters
   mdf(i0, a0, r)          "CO2 emissions from domestic intermediate demand"
   mmf(i0, a0, r)          "CO2 emissions from domestic intermediate demand"
   mdp(i0, r)              "CO2 emissions from domestic private demand"
   mmp(i0, r)              "CO2 emissions from domestic private demand"
   mdg(i0, r)              "CO2 emissions from domestic public demand"
   mmg(i0, r)              "CO2 emissions from domestic public demand"
   mdi(i0, r)              "CO2 emissions from investment demand"
   mmi(i0, r)              "CO2 emissions from investment demand"
;

$ifthen exist "%inDir%/%BaseName%Emiss.gdx"
   $gdxin "%inDir%/%BaseName%Emiss.gdx"
   $load mdf mmf mdp mmp mdg mmg mdi mmi
   $gdxin
$else
   mdf(i0,a0,r) = 0 ;
   mmf(i0,a0,r) = 0 ;
   mdp(i0,r)    = 0 ;
   mmp(i0,r)    = 0 ;
   mdg(i0,r)    = 0 ;
   mmg(i0,r)    = 0 ;
   mdi(i0,r)    = 0 ;
   mmi(i0,r)    = 0 ;
$endif

* --------------------------------------------------------------------------------------------------
*
*     Read in the GTAP Parameters
*
* --------------------------------------------------------------------------------------------------

Parameters
   esubt(a0,r)       "Top level CES substitution elasticity"
   esubc(a0,r)       "ND nest CES substitution elasticity"
   esubva(a0,r)      "VA nest CES substitution elasticity"

   etraq(a0,r)       "CET make elasticity"
   esubq(i0,r)       "CES make elasticity"

   incpar(i0,r)      "CDE expansion parameter"
   subpar(i0,r)      "CDE substitution parameter"

   esubg(r)          "CES government expenditure elasticity"
   esubi(r)          "CES investment expenditure elasticity"

   esubd(i0,r)       "Top level Armington elasticity"
   esubm(i0,r)       "Second level Armington elasticity"
   esubs(i0)         "CES margin elasticity"

   etrae(fp,r)       "CET elasticity for factors"
   rorFlex0(r)       "Flexibility of foreign capital"
;

$gdxin "%inDir%/%BaseName%Prm.gdx"
$loadDC esubt=esubt esubc=esubc esubva=esubva
$loadDC etraq=etraq esubq=esubq
$loadDC incpar=incpar subpar=subpar esubg=esubg esubi=esubi
$loadDC esubd=esubd esubm=esubm esubs=esubs
$loadDC etrae=etrae rorFlex0=rorFlex
$gdxin

* --------------------------------------------------------------------------------------------------
*
*     Declare and initialize the model parameters--overrides can be inserted before 'cal.gms'
*
* --------------------------------------------------------------------------------------------------

Parameters

*  Parameters normally sourced from GTAP

   sigmap(r,a)       "Top level CES production elasticity (ND/VA)"
   sigmand(r,a)      "CES elasticity across intermediate inputs"
   sigmav(r,a)       "CES elasticity across factors of production"

   omegas(r,a)       "Commodity supply CET elasticity"
   sigmas(r,i)       "Commodity supply CES elasticity"

   eh0(r,i)          "CDE expansion parameter"
   bh0(r,i)          "CDE substitution parameter"

   sigmag(r)         "CES government expenditure elasticity"
   sigmai(r)         "CES investment expenditure elasticity"

   sigmam(r,i,aa)    "Top level Armington elasticity"
   sigmaw(r,i)       "Second level Armington elasticity"
   sigmamg(i)        "CES expenditure elasticity for margin services exports"

   omegaf(r,fp)      "CET mobility elasticity for mobile factors"
   rorFlex(r,t)      "Flexibility of foreign capital"

*  Parameters in addition to standard GTAP model

   omegax(r,i)       "Top level output CET elasticity"
   omegaw(r,i)       "Second level export CET elasticity"

   etaf(r,fp)        "Aggregate factor supply elasticity"
   etaff(r,fp,a)     "Sector specific supply elasticity for non-mobile factors"

   mdtx0(r)          "Initial marginal tax rate"
   RoRFlag           "Capital account closure flag"
;

*  Overrides for GTAP-based parameters
*  If no overrides, parameters will be set in 'cal.gms'

sigmap(r,a)    = na ;
sigmand(r,a)   = na ;
sigmav(r,a)    = na ;

*  !!!! Explicitly assumes that these are not aggregated

omegas(r,a)    = -etraq(a, r) ;
sigmas(r,i)    = inf$(esubq(i,r) eq 0)
               + (1/esubq(i,r))$(esubq(i,r) ne 0)
               ;

eh0(r,i)       = na ;
bh0(r,i)       = na ;

sigmag(r)      = na ;
sigmai(r)      = na ;

sigmam(r,i,aa) = na ;
sigmaw(r,i)    = na ;
sigmamg(i)     = na ;

loop(fm,
   loop(endwm,
      omegaf(r,fm)$sameas(endwm,fm) = inf ;
   ) ;
   loop(endws,
      omegaf(r,fm)$sameas(endws,fm) = na ;
   ) ;
) ;
rorFlex(r,t)   = rorFlex0(r) ;

*  Other initialization -- use default GTAP assumptions

omegax(r,i)   = inf ;
omegaw(r,i)   = inf ;

etaf(r,fm)    = 0 ;
etaff(r,fp,a) = 0 ;

mdtx0(r)      = na ;
