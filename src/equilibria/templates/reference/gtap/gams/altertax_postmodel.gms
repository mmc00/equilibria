********************************************************************************
$ontext

   CGEBOX project

   GAMS file : ALTERTAX.GMS

   @purpose  : Output a new data base built from model results
   @author   : W.Britz
   @date     : 25.10.16
   @since    :
   @refDoc   :
   @seeAlso  :
   @calledBy :

$offtext
********************************************************************************
*
* --- load AEZ symbols not used or deleted
*
  $$iftheni.aezFound "%sysEnv.aez%"=="found"

     $$ifthen.aez defined xaez
        set land_cover_ori;
           $$GDXIN %resDir%/build/%dataSet%
           $$LOAD land_cover_ori=land_cover
           $$GDXIN
        $$ifthen.raez not defined rAez

*          --- load intern AEZ codes
*
           set rAez;
           $$GDXIN %resDir%/build/%dataSet%
           $$LOAD rAez
           $$GDXIN
*
*          --- and their link to regions and aezs
*
           set rAez_r_aez(raez,r,Aez);
           $$GDXIN %resDir%/build/%dataSet%
           $$LOAD rAez_r_aez
           $$GDXIN

        $$endif.raez
     $$else.aez
        set aez,raez,rAez_r_aez,land_cover;
        alias(land_cover,land_cover_ori);
        parameter p_landUse,p_carbon;
        $$GDXIN %resdir%/build/%dataset%.gdx
        $$LOAD AEZ,p_landUse=landUse,rAez,raez_r_aez,land_cover,p_carbon
        $$GDXIN
        set unManaged(land_cover) "Land not in economic use" /  SavnGrasslnd,Shrubland,Otherland/;
     $$endif.aez
     set aezFlagNew(*,aez);
     parameter p_landUseOri(*,*,*,*);
  $$endif.aezFound
*
* --- load CC damages if in DB, but not active
*
  $$iftheni.ccDamagesFound "%sysEnv.ccDamages%"=="found"
     $$ifthen.noDef not defined p_ccDamages
         parameter p_ccDamages;
         $$GDXIN %resdir%/build/%dataset%.gdx
           $$LOAD p_ccDamages
         $$GDXIN
     $$endif.noDef
  $$endif.ccDamagesFound
*
* --- load MTOE emission factors if in DB, but not active
*
  $$iftheni.mtoeFound "%sysEnv.mtoe%"=="found"
     $$ifthen.noDef not defined emtoeI0
         parameter emtoeI0,emtoeD0,emtoeA0;
         $$GDXIN %resdir%/build/%dataset%.gdx
           $$LOAD emtoeI0,emtoeD0,emtoeA0
         $$GDXIN
     $$endif.noDef
  $$endif.mtoeFound
*
* --- load MRIO split factor if in DB, but not active
*
  $$iftheni.mrioFound "%sysEnv.mrioSplitFactors%"=="found"
     $$ifthen.mrio not defined mrioSplitFactors
       $$if not defined mrioA set mrioA / int,hhsld,gov,inv/;
       set priceLevels / cif,m /;
       parameter mrioSplitFactors(r,i,r,mrioA,priceLevels)
       execute_load "%resdir%/build/%dataset%.gdx"  mrioSplitFactors;
     $$endif.mrio
  $$endif.mrioFound
*
* --- load air emission factors if in DB, but not active
*
  $$iftheni.airPollutFound "%sysEnv.emisBst%"=="found"
     $$ifthen.airPollut not defined emisbst
       set emisBst;
       $$GDXIN %resdir%/build/%dataset%.gdx
          $$LOAD emisBst
       $$GDXIN
       parameter emiiA(r,emisBst,*,aa)
       execute_load "%resdir%/build/%dataset%.gdx" emiiA;
     $$endif.airpollut
  $$endif.airPollutFound
*
* --- load non-CO2 emission factors if in DB, but not active
*
  $$iftheni.nonCo2Found "%sysEnv.nCO2%"=="found"
     $$iftheni.nonco2 not defined nCO2

        set nCO2;
        $$GDXIN %resdir%/build/%dataset%.gdx
           $$LOAD nCO2
        $$GDXIN
        parameter emiiN0(r,nCo2,*,aa);
        execute_load "%resdir%/build/%dataset%.gdx" emiiN0=emiiN;
        set nCO2r(*) "Non co2-emissions for reporting";nCo2R(nCO2) = YES;nCO2R("totCO2") = YES;
     $$endif.nonco2
  $$endif.nonCo2Found

  $$ifthen.co2 not defined emid0

     parameter
        emid0(r,i,aa)     "CO2 emission coefficient from domestic consumption"
        emii0(r,i,aa)     "CO2 emission coefficient from imported consumption"
     ;
     execute_load "%resdir%/build/%dataset%.gdx"  emid0,emii0;

  $$endif.co2
*
* --- load GMIG data if in DB, but not active
*
  $$iftheni.nonGIGFound "%sysEnv.p_remit%"=="found"
     $$ifthen.GGIG not defined p_remit
        parameter p_labMig(rNat,rNat,f)
                  p_popMig(rNat,rNat)
                  p_xftMig(rNat,rNat,f)
                  p_kappaMig(rNat,rNat,f)
                  p_fcttxMig(rNat,rNat,f)
                  p_remit(rnat,rnat,f)
        ;
        $$GDXIN %resdir%/build/%dataset%.gdx
           $$LOAD  p_labMig,p_popMig,p_kappaMig,p_fcttxMig,p_xftMig,p_remit
        $$GDXIN
     $$endif.GGIG
  $$endif.nonGIGFound
*
* --- load FABIO declaration, if not yet available
*
*
* --- load original mapping of commodities
*
  set mapiOri,mapaOri;
  $$GDXIN %resdir%/build/%dataset%.gdx
    $$LOAD mapiOri=mapi
  $$GDXIN

  set hOut / hhsld "Household" /;
  $$onmulti
  set isOut / set.is/;
  set isOut / set.hOut/;
  set aaOut / set.aa/;
  set aaOut / set.hout/;
  $$offmulti

  isOut(h) $ (not sameas(h,"hhsld")) = no;
  aaOut(h) $ (not sameas(h,"hhsld")) = no;

  set capOut / set.cap/;
  capOut("oldCap") = Yes;
  capOut("newCap") = Yes;

  parameter p_capTransB(rNat)
            p_capTrans(r,t);
  scalar altertax / 1 /;

  $$iftheni.dyn "%dynMode%"=="Recursive dynamic"
*
*    --- in rec-dyn, we populate the same for multiple time points
*        and generate a GDX container for each one
*
     $$setglobal tIndex tRep
     loop(tRep,
  $$else.dyn

     $$setglobal tIndex "'shock'"

  $$endif.dyn

*    --- Save the data for current year

     sam0(rSamReg,is,js) = sam(rSamReg,is,js,%tIndex%);
*
*    --- all the following positions would need to be updated based on simulated
*        results. Attention: everthing must be expressed in world curreny values again!
*

*    --- scale back Armington demand at agent prices

     xda0(r,i,aa)     $ xd.l(r,i,aa,%tIndex%)  = max(0,xd.l(r,i,aa,%tIndex%)) * pdp.l(r,i,aa,%tIndex%) / gblscale;
     xma0(r,i,aa)     $ xm.l(r,i,aa,%tIndex%)  = max(0,xm.l(r,i,aa,%tIndex%)) * pmp.l(r,i,aa,%tIndex%) / gblscale;

     xda0(r,i,"hhsld") $ sum(h $ (not sameas(h,"hhsld")), yc.l(r,h,%tindex%)) = 0;
     xma0(r,i,"hhsld") $ sum(h $ (not sameas(h,"hhsld")), yc.l(r,h,%tindex%)) = 0;

     xda0(rnat,i,"trdMg") = 0;
*
*    --- remove results for aggregated Armington agents, should not be part of new DB
*
     $$ifi "%modulesAggregate_Armington%" =="ON"   xda0(r,i,"armA") = 0;
     $$ifi "%modulesAggregate_Armington%" =="ON"   xma0(r,i,"armA") = 0;
     $$ifi "%modulesAggregate_firm_demand%" =="ON" xda0(r,i,"int") = 0;
     $$ifi "%modulesAggregate_firm_demand%" =="ON" xma0(r,i,"int") = 0;
*
*    --- remove results for multiple households, not part of standard DB
*
     xda0(r,i,"hhsld")  = sum(h, xda0(r,i,h));
     xma0(r,i,"hhsld")  = sum(h, xma0(r,i,h));

     xda0(r,i,h) $ (not sameas(h,"hhsld")) = 0;
     xma0(r,i,h) $ (not sameas(h,"hhsld")) = 0;
*
*    --- remove results for aggregated Armington regions, should not be part of new DB
*
     $$ifi "%modulesAggregate_Armington%" =="ON"   xdm0(r,i,"armA") = 0;
     $$ifi "%modulesAggregate_Armington%" =="ON"   xmm0(r,i,"armA") = 0;
     $$ifi "%modulesAggregate_firm_demand%" =="ON" xdm0(r,i,"int") = 0;
     $$ifi "%modulesAggregate_firm_demand%" =="ON" xmm0(r,i,"int") = 0;
*
*    --- calculate results at market prices
*
     xdm0(r,i,aa)     $ xd.l(r,i,aa,%tIndex%)  = max(0,xd.l(r,i,aa,%tIndex%)) * pd.l(r,i,%tIndex%)/gblScale;
     xmm0(r,i,aa)     $ xm.l(r,i,aa,%tIndex%)  = max(0,xm.l(r,i,aa,%tIndex%)) * pmt.l(r,i,%tIndex%)/gblScale;
*
*    --- remove results for multiple households, not part of standard DB
*
     xdm0(r,i,"hhsld") $ sum(h $ (not sameas(h,"hhsld")), yc.l(r,h,%tindex%)) = 0;
     xmm0(r,i,"hhsld") $ sum(h $ (not sameas(h,"hhsld")), yc.l(r,h,%tindex%)) = 0;

     xdm0(r,i,"hhsld") = sum(h, xdm0(r,i,h));
     xmm0(r,i,"hhsld") = sum(h, xmm0(r,i,h));

     xdm0(r,i,h) $ (not sameas(h,"hhsld")) = 0;
     xmm0(r,i,h) $ (not sameas(h,"hhsld")) = 0;
     xdm0(rnat,i,"trdMg")                  = 0;

     $$iftheni.MACCs "%MACCsProcessEmis%"=="on"
         xdm0(r,i,gov) =  xdm0(r,i,gov) + xdm0(r,i,"maccs");
         xda0(r,i,gov) =  xda0(r,i,gov) + xda0(r,i,"maccs");
         xmm0(r,i,gov) =  xmm0(r,i,gov) + xmm0(r,i,"maccs");
         xma0(r,i,gov) =  xma0(r,i,gov) + xma0(r,i,"maccs");

         xdm0(r,i,"maccs") = 0;
         xda0(r,i,"maccs") = 0;
         xmm0(r,i,"maccs") = 0;
         xma0(r,i,"maccs") = 0;
     $$endif.MACCs

*
*    --- margin demand (substituted out in model, recalculate)
*
     xmarg0(m,rNat,i,rpNat) $ p_amgm(m,rNat,i,rpNat)
        = tmarg(rNat,i,rpNat,%tIndex%) * xw.l(rNat,i,rpNat,%tIndex%)
                                       * p_amgm(m,rNat,i,rpNat)/m_lambdamg(m,rNat,i,rpNat,%tIndex%) * ptmg.l(m,%tIndex%) / gblScale;
*
*    --- capital stocks, depreciations, import, export and factor taxes
*

     kstock0(r)       $ kstock0(r)       = kstock.l(r,%tIndex%) * 1/gblScale;
     depry0(r)        $ depry0(r)        = p_fdepr(r,%tIndex%) * pi.l(r,%tIndex%) * kstock.l(r,%tIndex%) / gblscale;
     imptxY0(rNat,rpNat,i)  = imptx.l(rNat,i,rpNat,%tIndex%) * pmcif.l(rNat,i,rpNat,%tIndex%) * xw.l(rNat,i,rpNat,%tIndex%) / gblScale;
     exptxY0(rNat,rpNat,i)  = exptx.l(rNat,i,rpNat,%tIndex%) * pe.l(rNat,i,rpNat,%tIndex%)    * xw.l(rNat,i,rpNat,%tIndex%) / gblScale;
     fcttsY0(r,f,a)   = fctts.l(r,f,a,%tIndex%) * sam(r,f,a,%tIndex%);

     fcttxY0(r,f,a)   = fcttx.l(r,f,a,%tIndex%) * sam(r,f,a,%tIndex%)
          $$iftheni.CO2 "%ModulesCO2_Emissions%"=="on"
                  + sum(rr_rr(r,rNat),emisp(rNat,%tindex%) * emifCo2eqTaxed(rNat,f,a,%tindex%)
                            *(1-v_redEmiaCo2Eq(rNat,a,%tindex%) + p_emiShift(rNat,f,a,%tindex%))*xf.l(r,f,a,%tindex%)/ gblScale

                $$ifi "%modulesGTAP_AEZ%"=="on"                  + emispLnd(rNat,a,%tindex%)/ gblScale*xf.l(r,f,a,%tindex%) $ lnd(f)
                                                  )
             $$endif.CO2
      ;

     $$iftheni.co2eq "%Co2eq%"=="on"
*
*      --- rebase emission factors (necessary if MTOE is active)
*

       emid(r,i,"hhsld",%tindex%) = smax(h,emid(r,i,h,%tindex%));
       emii(r,i,"hhsld",%tindex%) = smax(h,emii(r,i,h,%tindex%));

       emid0(r,i,aa) $ xda0(r,i,aa)  = emid(r,i,aa,%tindex%) / ps.l(r,i,"%t0%");
       emii0(r,i,aa) $ xma0(r,i,aa)  = emii(r,i,aa,%tindex%) / pmt.l(r,i,"%t0%");

       emid0(r,i,"armA")  = 0;
       emii0(r,i,"armA")  = 0;

       $$iftheni.non_co2 "%ModulesNON_CO2_Emissions%"=="on"

          emiiN(r,nco2,i,"hhsld",%tindex%) = smax(h,emiiN(r,nCo2,i,h,%tindex%));
          emiin0(r,nco2,i,aa)              = emiiN(r,nco2,i,aa,%tindex%);
          emiin0(r,nco2,i,"armA")          = 0;
          emiin0(r,nco2,"",a)              = emiiN(r,nco2,"",a,%tindex%);
          emiin0(r,nco2,f,a)               = emiiN(r,nco2,f,a,%tindex%);
       $$endif.non_co2

     $$endif.co2eq

     p_capTransB(rNat)   = p_capTrans(rNat,%tIndex%);
*
*    --- NUTS2-extension switched off: load sub regional land use data and apply
*        proportionality assumptions
*
     execute_load "%resdir%/build/%dataset%.gdx" p_landUseOri=landUse;

     $$ifthen.aez defined xaez

        option kill=p_landUse;

        $$ifi defined aezFlagOri  aezFlag(r,aez) $ aezFlagOri(r,aez) = YES;
        $$ifi defined aezFlagOri  aezFlag(r,"aezAgg") = No;

        $$iftheni.aezAgg "%AezAgg%"=="ON"
           $$ifi not "%outputtypesGUI%"=="ON" $$abort Please switch on output to GUI for ALTERTAX is AEZ are aggregated into
        $$endif.aezAgg
*
*       --- update land use ori with correction in GTAP-AEZ cal. This is importing for the proportionality
*           assumptions below for disrNew

        p_landUseOri(aezFlag(r,aez),a,"vfm")        = xAez.l(r,aez,a,"%t0%") * pAez.l(r,aez,a,"%t0%") / gblScale;
        p_landUseOri(aezFlag(r,aez),aezA,"ha")      = xAez.l(r,aez,aezA,"%t0%") / gblScale * 1000;
        p_landUseOri(aezFlag(r,aez),aezA,"rent")    = pAez.l(r,aez,aezA,"%t0%");
        p_landUseOri(aezFlag(r,aez),unmanaged,"ha") = unManagedLnd.l(r,aez,unManaged,"%t0%") / gblScale * 1000;
        p_landUseOri(aezFlag(r,aez),landCat,"ha")   = sum(landCat_a(landCat,aezA), p_landUseOri(r,aez,aezA,"ha"));
        p_landUseOri(aezFlag(r,aez),landCat,"vfm")  = sum(landCat_a(landCat,aezA), p_landUseOri(r,aez,aezA,"vfm"));
        p_landUseOri(aezFlag(r,aez),"Land","vom")   = sum(aezA, p_landUseOri(r,aez,aezA,"vfm"));

        p_landUse(aezFlag(r,aez),aezA,"vfm")         = xAez.l(r,aez,aezA,%tIndex%) * pAez.l(r,aez,aezA,%tIndex%) / gblScale;
        p_landUse(aezFlag(r,aez),aezA,"ha")          = xAez.l(r,aez,aezA,%tIndex%) / gblScale * 1000;
        p_landUse(aezFlag(r,aez),aezA,"rent")        = pAez.l(r,aez,aezA,%tIndex%);
        p_landUse(aezFlag(r,aez),unmanaged,"ha")     = unManagedLnd.l(r,aez,unManaged,%tIndex%) / gblScale * 1000;
        p_landUse(aezFlag(r,aez),landCat,"ha")       = sum(landCat_a(landCat,aezA),  p_landUse(r,aez,aezA,"ha"));
        p_landUse(aezFlag(r,aez),landCat,"vfm")      = sum(landCat_a(landCat,aezA),  p_landUse(r,aez,aezA,"vfm"));
        p_landUse(aezFlag(r,aez),"Land","vom")       = sum(aezA, p_landUse(r,aez,aezA,"vfm"));
        p_landUse(aezFlag(r,aez),"builtUpLand","ha") = p_landUseOri(r,aez,"builtUpLand","ha") * (1 + p_builtupLandChange(r,%tindex%));

        p_carbon(r,"",land_cover,"carbon") = 0;

        if ( card(subr),

           aezFlag(disr,aez) $ sum(r_r(subr,disr),aezFlag(subr,aez)) = YES;

           p_landUse(aezFlag(disr,aez),aezA,"vfm")      = sum(r_r(subr,disr),   p_landUse(subr,aez,aezA,"vfm"));
           p_landUse(aezFlag(disr,aez),aezA,"ha")       = sum(r_r(subr,disr),   p_landUse(subr,aez,aezA,"ha"));
           p_landUse(aezFlag(disr,aez),unmanaged,"ha")  = sum(r_r(subr,disr),   p_landUse(subr,aez,unmanaged,"ha"));
           p_landUse(aezFlag(disr,aez),landCat,"vfm")   = sum(r_r(subr,disr),   p_landUse(subr,aez,landCat,"vfm"));
           p_landUse(aezFlag(disr,aez),land_cover,"ha") = sum(r_r(subr,disr),   p_landUse(subr,aez,land_cover,"ha"));
           p_landUse(aezFlag(disr,aez),"land","vom")    = sum(r_r(subr,disr),   p_landUse(subr,aez,"land","vom"));

        );

        p_landUse(aezFlag(r,aez),landCat,"rent") $ p_landUse(r,aez,landCat,"ha")
         = p_landUse(r,aez,landCat,"vfm") / p_landUse(r,aez,landCat,"ha") * 1000;
        p_landUse(aezFlag(r,aez),"tot","ha")         = sum(land_cover,p_landUse(r,aez,land_cover,"ha"));
        subrNew(subr) = no;

     $$endif.aez

     aezFlagNew(subrNew,aez) $ p_landUseOri(subrNew,aez,"Land","vom") = yes;

     p_landUseOri(disrNew,aez,a,"ha")  $ (p_landUseOri(disrNew,aez,a,"ha") lt 1.E-3) = 0;
     p_landUseOri(disrNew,aez,a,"ha")  $ (not p_landUseOri(disrNew,aez,a,"vfm"))     = 0;


     p_landUse(aezFlagNew(subrNew,aez),a,"vfm") $ sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,a,"vfm"))
      = p_landUseOri(subrNew,aez,a,"vfm") * sum(subReg_r(subrNew,disrNew),p_landuse(disrNew,aez,a,"vfm")/p_landUseOri(disrNew,aez,a,"vfm"));

     p_landUse(aezFlagNew(subrNew,aez),a,"vfm") $ (not sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,a,"vfm"))) = 0;

     p_landUse(aezFlagNew(subrNew,aez),a,"ha") $ sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,a,"ha"))
      = p_landUseOri(subrNew,aez,a,"ha") * sum(subReg_r(subrNew,disrNew),p_landuse(disrNew,aez,a,"ha")/p_landUseOri(disrNew,aez,a,"ha"));

     p_landUse(aezFlagNew(subrNew,aez),a,"ha")  $ (not sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,a,"ha"))) = 0;

     p_landUse(aezFlagNew(subrNew,aez),land_cover,"vfm") $ sum(subReg_r(subrNew,disrNew),p_landuseOri(disrNew,aez,land_cover,"vfm"))
      = p_landUseOri(subrNew,aez,land_cover,"vfm")
        * sum(subReg_r(subrNew,disrNew),p_landuse(disrNew,aez,land_cover,"vfm")/p_landUseOri(disrNew,aez,land_cover,"vfm"));

     p_landUse(aezFlagNew(subrNew,aez),land_cover,"vfm") $ (not sum(subReg_r(subrNew,disrNew),p_landuseOri(disrNew,aez,land_cover,"vfm"))) = 0;

     p_landUse(aezFlagNew(subrNew,aez),land_cover,"ha") $ sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,land_cover,"ha"))
      = p_landUseOri(subrNew,aez,land_cover,"ha")
        * sum(subReg_r(subrNew,disrNew),p_landuse(disrNew,aez,land_cover,"ha")/p_landUseOri(disrNew,aez,land_cover,"ha"));

     p_landUse(aezFlagNew(subrNew,aez),land_cover,"ha") $ (not sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,land_cover,"ha"))) = 0;

     p_landUse(aezFlagNew(subrNew,aez),unManaged,"ha") $ sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,unManaged,"ha"))
      = p_landUseOri(subrNew,aez,unManaged,"ha")
         * sum(subReg_r(subrNew,disrNew),p_landuse(disrNew,aez,unManaged,"ha")/p_landUseOri(disrNew,aez,unManaged,"ha"));

     p_landUse(aezFlagNew(subrNew,aez),unManaged,"ha") $ (not sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,unManaged,"ha"))) = 0;

     p_landUse(aezFlagNew(subrNew,aez),"Land","vom") $ sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,"Land","vom"))
      = p_landUseOri(subrNew,aez,"Land","vom")
       * sum(subReg_r(subrNew,disrNew),p_landuse(disrNew,aez,"Land","vom")/p_landUseOri(disrNew,aez,"Land","vom"));

     p_landUse(aezFlagNew(subrNew,aez),"Land","vom") $ (not sum(subReg_r(subrNew,disrNew),p_landUseOri(disrNew,aez,"Land","vom"))) = 0;

     subrNew(subr) = yes;

     $$Ifthen.fabio defined p_nutContFabio

        p_nutContFabio(rnat,i,"xa",%tindex%)
          =          xa.l(rnat,i,"hhsld",%tIndex%)*pa.l(rnat,i,"hhsld",%tIndex%)
            +sum(gov,xa.l(rnat,i,gov,%tIndex%)    *pa.l(rnat,i,gov,%tIndex%))
            +sum(inv,xa.l(rnat,i,inv,%tIndex%)    *pa.l(rnat,i,inv,%tIndex%));

        p_nutContFabioIn(rnat,"total",nut) = sum( i $ p_nutContFabio(rnat,i,nut,%tindex%),   p_nutContFabio(rnat,i,nut,%tindex%)
                                                                                           * p_nutContFabio(rnat,i,"xa",%tindex%));

        p_nutContFabioIn(rnat,i,nut)         = p_nutContFabio(rnat,i,nut,%tindex%);
        p_nutContFabioIn(rnat,i,"xa")        = p_nutContFabio(rnat,i,"xa",%tindex%);
        p_nutContFabioIn(rnat,"total","xa")  = 0;

     $$endif.fabio
*
*    --- store scale from GDP at market prices to real GDP. Important to show
*        (1) Real GDP level in derived simulations, (2) cardinal utility estimates in AIDADS,
*        (3) Calorie intake estimates from real GDP per capita
*
     p_scaleRGdpMp(rnat) = 1/pgdpmp.l(rnat,%tIndex%);

     $$iftheni.dyn "%dynMode%"=="Recursive dynamic"
        put_utilities batch 'gdxout' / '%resdir%/build/%dataset%%postfix%_alttax_' tRep.tl:0  '.gdx';
        execute_unload
     $$else.dyn
        execute_unload "%resdir%/build/%dataset%%postfix%_alttax.gdx"
     $$endif.dyn

        isOut=is, aaOut=aa, a, i, rSamReg=r, f, l, capout=cap, lnd, nr, fd, hout=h, gov, inv,isall,disrAll=disr
        fDat,rDat,iDat,subrNew=subr,rNat,subReg_r
        sam0, xda0, xdm0, xma0, xmm0, xmarg0,
        eh0, bh0, sigmap0, sigmav0, sigmai0, sigmam0, sigmaw0, omegaf0, RoRFlex0,
        kstock0, depry0, pop0, p_scaleRGdpMp
        emid0,emii0,
        exptxY0, imptxY0, fcttxY0, fcttsY0
        mapiOri=mapi,mapr=rr,m,mapa=aa1,ff,
        mapi=ii,mapr=rr,m,mapa=aa1,ff,
        $$iftheni.aez defined aez
           AEZ,p_landuse=landUse,rAez,raez_r_aez,land_cover_ori=land_cover,p_carbon
        $$endif.aez
        $$ifthen defined nCO2
           nCO2,emiiN0=emiiN
        $$endif
        $$ifthen defined emisbst
           emisBst,emiiA,
        $$endif
        $$ifthen defined mrioSplitFactors
           mrioSplitFactors
        $$endif
        $$ifthen defined p_labMig
            p_labMig,p_popMig
        $$endif
        $$ifthen defined p_kappaMig
            p_kappaMig,p_fcttxMig,p_xftMig,p_remit
        $$endif
        $$ifthen defined p_ccDamages
           p_ccDamages
        $$endif
        $$ifthen defined emtoeI0
           emtoeI0,emtoeD0,emtoeA0
        $$endif
        $$Ifthen defined p_nutContFabio
           p_nutContFabioIn=p_nutContFabio
        $$endif

        p_capTransB

*       $$ifi "%filterMethod%"=="Rebalancing" itrlog,keepCor.l
*       $$ifi "%loadLandUse%"=="on" landUse,AEZ,land_cover
        $$ifthen defined s_meta
           s_meta
        $$endif
        altertax
     ;

 $$iftheni.dyn "%dynMode%"=="Recursive dynamic"
*
*   --- end of t loop
*
    );
 $$endif.dyn
 $$if not errorfree $abort Compilation error after file: %system.fn%
