*==============================================================================*
*                              CALIBRATION MODULE                             *
*==============================================================================*
*                                                                              *
*  This module contains all calibration calculations for the PEP CGE model    *
*  including income calculations, tax rates, functional form parameters,      *
*  and benchmark GDP measures                                                  *
*                                                                              *
*==============================================================================*

* 4 Calibration

** 4.1 Calculation of income and savings related variables and parameters

*** 4.1.1 Household income components
 YHKO(h)         = SUM[k,lambda_RK(h,k)];
 YHLO(h)         = SUM[l,lambda_WL(h,l)];
 YHTRO(h)        = SUM[ag,TRO(h,ag)];
 YHO(h)          = YHLO(h)+YHKO(h)+YHTRO(h);
 YDHO(h)         = YHO(h)-TDHO(h)-TRO('gvt',h);
 CTHO(h)         = YDHO(h)-SHO(h)-SUM[agng,TRO(agng,h)];

*** 4.1.2 Firm income components
 YFKO(f)         = SUM[k,lambda_RK(f,k)];
 YFTRO(f)        = SUM[ag,TRO(f,ag)];
 YFO(f)          = YFKO(f)+YFTRO(f);
 YDFO(f)         = YFO(f)-TDFO(f);

*** 4.1.3 Government income and tax aggregates
 YGKO            = SUM[k,lambda_RK('gvt',k)];
 TDHTO           = SUM[h,TDHO(h)];
 TDFTO           = SUM[f,TDFO(f)];
 TICTO           = SUM[i,TICO(i)];
 TIMTO           = SUM[i,TIMO(i)];
 TIXTO           = SUM[i,TIXO(i)];
 TIWTO           = SUM[(l,j),TIWO(l,j)];
 TIKTO           = SUM[(k,j),TIKO(k,j)];
 TIPTO           = SUM[j,TIPO(j)];
 TPRODNO         = TIKTO+TIWTO+TIPTO;
 TPRCTSO         = TICTO+TIMTO+TIXTO;
 YGTRO           = SUM[ag,TRO('gvt',ag)];
 YGO             = YGKO+TDHTO+TDFTO+TPRODNO+TPRCTSO+YGTRO;

*** 4.1.4 Rest-of-world and investment accounts
 YROWO           = SUM[i,IMO(i)]+SUM[k,lambda_RK('row',k)]
                  +SUM[ag,TRO('row',ag)];
 CABO            = -SROWO;
 ITO             = SUM[h,SHO(h)]+SUM[f,SFO(f)]+SGO+SROWO;

*** 4.1.5 Income distribution and transfer parameters
 lambda_RK(ag,k) = lambda_RK(ag,k)/SUM[j,KDO(k,j)];
 lambda_WL(h,l)  = lambda_WL(h,l)/SUM[j,LDO(l,j)];
 lambda_TR(agng,h)
                 = TRO(agng,h)/YDHO(h);
 lambda_TR(ag,f) = TRO(ag,f)/YDFO(f);

*** 4.1.6 Savings and transfer function slopes
 sh1O(h)         = [SHO(h)-sh0O(h)]/YDHO(h);
 tr1O(h)         = [TRO('gvt',h)-tr0O(h)]/YHO(h);

** 4.2 Calibration of investment and government spending shares

 gamma_GVT(i)    = CGO(i)/SUM[ij,CGO(ij)];
 gamma_INV(i)    = INVO(i)/SUM[ij,INVO(ij)];

** 4.3 Calibration of direct tax rates

 ttdf1O(f)       = [TDFO(f)-ttdf0O(f)]/YFKO(f);
 ttdh1O(h)       = [TDHO(h)-ttdh0O(h)]/YHO(h);

** 4.4 Calibration of other tax rates, prices, margins and volumes

*** 4.4.1 Price calibration and margin rates
 PCO(i)          = [DDO(i)+IMO(i)+SUM(ij,tmrg(ij,i))+TICO(i)+TIMO(i)]
                  /[DDO(i)+IMO(i)];
 tmrg(i,ij)      = tmrg(i,ij)/PCO(i);
 tmrg_X(i,ij)    = tmrg_X(i,ij)/PCO(i);

*** 4.4.2 Volume adjustment from prices
 DDO(i)          = DDO(i)/PLO(i);
 IMO(i)          = IMO(i)/(PWMO(i)*eO);

*** 4.4.3 Margin and tax rate calculations
 tmrg(i,ij)      = tmrg(i,ij)/{DDO(ij)+IMO(ij)};

 tticO(i)        = TICO(i)/{(PLO(i)+SUM[ij,PCO(ij)*tmrg(ij,i)])*DDO(i)
                           +(eO*PWMO(i)+SUM[ij,PCO(ij)*tmrg(ij,i)])*IMO(i)
                           +TIMO(i)};

 PDO(i)          = {PLO(i)+SUM[ij,PCO(ij)*tmrg(ij,i)]}*(1+tticO(i));

 ttimO(i)$IMO(i) = TIMO(i)/[eO*PWMO(i)*IMO(i)];

 PMO(i)          = {(1+ttimO(i))*eO*PWMO(i)+SUM[ij,PCO(ij)*tmrg(ij,i)]}
                   *(1+tticO(i));

*** 4.4.4 Export price and tax calculations
 EXO(j,i)        = EXO(j,i)/PEO(i);
 tmrg_X(ij,i)$EXDO(i)
                 = tmrg_X(ij,i)/SUM[j,EXO(j,i)];
 ttixO(i)$EXDO(i)
                 = TIXO(i)/[EXDO(i)-TIXO(i)];
 PE_FOBO(i)      = (1+ttixO(i))*(PEO(i)+SUM[ij,PCO(ij)*tmrg_X(ij,i)]);
 PWXO(i)         = PE_FOBO(i)/eO;
 EXDO(i)         = EXDO(i)/(PWXO(i)*eO);

*** 4.4.5 Production and supply calculations
 DSO(j,i)        = DSO(j,i)/PLO(i);
 XSO(j,i)        = DSO(j,i)+EXO(j,i);
 PO(j,i)$XSO(j,i)= [PLO(i)*DSO(j,i)+PEO(i)*EXO(j,i)]/XSO(j,i);
 XSTO(j)         = SUM[i,XSO(j,i)];
 PTO(j)          = SUM[i$XSO(j,i),PO(j,i)*XSO(j,i)]/XSTO(j);

*** 4.4.6 Composite commodity and margin calculations
 QO(i)           = [PMO(i)*IMO(i)+PDO(i)*DDO(i)]/PCO(i);

 MRGNO(i)        = SUM[ij,tmrg(i,ij)*DDO(ij)]+
                   SUM[ij,tmrg(i,ij)*IMO(ij)]+
                   SUM[(j,ij),tmrg_X(i,ij)*EXO(j,ij)];

*** 4.4.7 Consumption and intermediate demand volumes
 CO(i,h)         = CO(i,h)/PCO(i);
 CGO(i)          = CGO(i)/PCO(i);
 DIO(i,j)        = DIO(i,j)/PCO(i);
 INVO(i)         = INVO(i)/PCO(i);
 VSTKO(i)        = VSTKO(i)/PCO(i);
 GFCFO           = ITO-SUM[i,PCO(i)*VSTKO(i)];

*** 4.4.8 Intermediate consumption aggregates
 CIO(j)          = SUM[i,DIO(i,j)];
 DITO(i)         = SUM[j,DIO(i,j)];
 GO              = SUM[i,PCO(i)*CGO(i)];

*** 4.4.9 Price indices for intermediate consumption
 PCIO(j)         = SUM[i,PCO(i)*DIO(i,j)]/CIO(j);

*** 4.4.10 Factor price and tax rate calculations
 ttiwO(l,j)$LDO(l,j)
                 = TIWO(l,j)/LDO(l,j);
 WTIO(l,j)       = WO(l)*(1+ttiwO(l,j));
 ttikO(k,j)$KDO(k,j)
                 = TIKO(k,j)/KDO(k,j);
 RTIO(k,j)       = RO(k,j)*(1+ttikO(k,j));

*** 4.4.11 Factor demand volumes and composite factor prices
 LDO(l,j)        = LDO(l,j)/WO(l);
 LDCO(j)         = SUM[l,LDO(l,j)];
 LSO(l)          = SUM[j,LDO(l,j)];
 WCO(j)$LDCO(j)  = SUM[l,WTIO(l,j)*LDO(l,j)]/LDCO(j);

 KDO(k,j)        = KDO(k,j)/RO(k,j);
 KDCO(j)         = SUM[k,KDO(k,j)];
 KSO(k)          = SUM[j,KDO(k,j)];
 RCO(j)$KDCO(j)  = SUM[k,RTIO(k,j)*KDO(k,j)]/KDCO(j);

*** 4.4.12 Value added and production tax calculations
 VAO(j)          = LDCO(j)+KDCO(j);
 PVAO(j)         = [WCO(j)*LDCO(j)+RCO(j)*KDCO(j)]/VAO(j);

 ttipO(j)        = TIPO(j)/{PVAO(j)*VAO(j)+SUM[i,PCO(i)*DIO(i,j)]};

 PPO(j)          = PTO(j)/(1+ttipO(j));

*** 4.4.13 Price indices (set to 1 for initial calibration)
* PIXGDPO is tautologically equal to 1, based on its formula
 PIXGDPO         = 1;

* PIXCONO is tautologically equal to 1, based on its formula
 PIXCONO         = 1;

* PIXGVTO is tautologically equal to 1, based on its formula
 PIXGVTO         = 1;

* PIXINVO is tautologically equal to 1, based on its formula
 PIXINVO         = 1;

** 4.5 Calibration of indexed transfers and parameters

 TRO(agd,'row')  = TRO(agd,'row')/PIXCONO**eta;
 TRO(agng,'gvt') = TRO(agng,'gvt')/PIXCONO**eta;
 ttdf0O(f)       = ttdf0O(f)/PIXCONO**eta;
 ttdh0O(h)       = ttdh0O(h)/PIXCONO**eta;
 sh0O(h)         = sh0O(h)/PIXCONO**eta;
 tr0O(h)         = tr0O(h)/PIXCONO**eta;

** 4.6 Calibration of function parameters

*** 4.6.1 Leontief functions
 io(j)           = CIO(j)/XSTO(j);
 v(j)            = VAO(j)/XSTO(j);
 aij(i,j)        = DIO(i,j)/CIO(j);

*** 4.6.2 Calibration of CET parameters
**** 4.6.2.1 CET between commodities
 rho_XT(j)       = (1+sigma_XT(j))/sigma_XT(j);
 beta_XT(j,i)$XSO(j,i)
                 = PO(j,i)*XSO(j,i)**(1-rho_XT(j))/
                   SUM[ij$XSO(j,ij),PO(j,ij)*XSO(j,ij)**(1-rho_XT(j))];
 B_XT(j)         = XSTO(j)
                  /SUM[i$XSO(j,i),beta_XT(j,i)*XSO(j,i)**rho_XT(j)
                  ]**(1/rho_XT(j));

**** 4.6.2.2 CET between exports and local production
 rho_X(j,i)$[EXO(j,i) and DSO(j,i)]
                 = (1+sigma_X(j,i))/sigma_X(j,i);
 rho_X(j,i)$[(not EXO(j,i)) or (not DSO(j,i))]
                 = 1;
 beta_X(j,i)$XSO(j,i)
                 = PEO(i)*EXO(j,i)**(1-rho_X(j,i))/
                  [PEO(i)*EXO(j,i)**(1-rho_X(j,i))
                  +PLO(i)*DSO(j,i)**(1-rho_X(j,i))];
 B_X(j,i)$XSO(j,i)
                 = XSO(j,i)/
                 [beta_X(j,i)*EXO(j,i)**rho_X(j,i)+
                 (1-beta_X(j,i))*DSO(j,i)**rho_X(j,i)]
                 **(1/rho_X(j,i));

*** 4.6.3 Calibration of CES parameters
**** 4.6.3.1 Composite good
 rho_M(i)$[IMO(i) and DDO(i)]
                 = (1-sigma_M(i))/sigma_M(i);
 rho_M(i)$[(not IMO(i)) or (not DDO(i))]
                 = -1;
 beta_M(i)$QO(i) = PMO(i)*IMO(i)**(rho_M(i)+1)/
                  [PMO(i)*IMO(i)**(rho_M(i)+1)+
                   PDO(i)*DDO(i)**(rho_M(i)+1)];
 B_M(i)$QO(i)    = QO(i)
                   /[beta_M(i)*IMO(i)**(-rho_M(i))+
                   (1-beta_M(i))*DDO(i)**(-rho_M(i))
                   ]**(-1/rho_M(i));

**** 4.6.3.2 Composite capital
 rho_KD(j)$KDCO(j)
                 = (1-sigma_KD(j))/sigma_KD(j);
 beta_KD(k,j)$KDO(k,j)
                 = RTIO(k,j)*KDO(k,j)**(rho_KD(j)+1)/
                   SUM[kj,RTIO(kj,j)*KDO(kj,j)**(rho_KD(j)+1)];
 B_KD(j)$KDCO(j) = KDCO(j)
                  /SUM[k$KDO(k,j),beta_KD(k,j)*KDO(k,j)**(-rho_KD(j))
                  ]**(-1/rho_KD(j));

**** 4.6.3.3 Composite labor
 rho_LD(j)       = (1-sigma_LD(j))/sigma_LD(j);
 beta_LD(l,j)$LDO(l,j)
                 = WTIO(l,j)*LDO(l,j)**(rho_LD(j)+1)/
                   SUM[lj,WTIO(lj,j)*LDO(lj,j)**(rho_LD(j)+1)];
 B_LD(j)$LDCO(j) = LDCO(j)/SUM[l,beta_LD(l,j)$LDO(l,j)*LDO(l,j)**(-rho_LD(j))]
                            **(-1/rho_LD(j));

**** 4.6.3.4 Value added
 rho_VA(j)$[KDCO(j) and LDCO(j)]
                 = (1-sigma_VA(j))/sigma_VA(j);
 rho_VA(j)$[(not KDCO(j)) or (not LDCO(j))]
                 = -1;
 beta_VA(j)      = WCO(j)*LDCO(j)**(rho_VA(j)+1)/
                  [WCO(j)*LDCO(j)**(rho_VA(j)+1)+
                   RCO(j)*KDCO(j)**(rho_VA(j)+1)];
 B_VA(j)         = VAO(j)
                   /[beta_VA(j)*LDCO(j)**(-rho_VA(j))+
                   (1-beta_VA(j))*KDCO(j)**(-rho_VA(j))
                   ]**(-1/rho_VA(j));

*** 4.6.4 Calibration of LES parameters
*   As the assigned values of income elasticities may not result in
*   consumption shares that add up to 1, this first step
*   adjusts the elasticities proportionally

 sigma_Y(i,h)    = sigma_Y(i,h)*CTHO(h)/SUM[ij,sigma_Y(ij,h)*PCO(ij)*CO(ij,h)];
 gamma_LES(i,h)  = PCO(i)*CO(i,h)*sigma_Y(i,h)/CTHO(h);
 CMINO(i,h)      = CO(i,h)+gamma_LES(i,h)*[CTHO(h)/
                  (PCO(i)*frisch(h))];

** 4.7 Calibration of gross domestic products

 GDP_BPO         = SUM[j,PVAO(j)*VAO(j)]+TIPTO;
 GDP_MPO         = GDP_BPO+TPRCTSO;
 GDP_IBO         = SUM[(l,j),WO(l)*LDO(l,j)]+SUM[(k,j),RO(k,j)*KDO(k,j)]
                   +TPRODNO+TPRCTSO;
 GDP_FDO         = SUM{i,PCO(i)*(SUM[h,CO(i,h)]+CGO(i)+INVO(i)+VSTKO(i))}
                  +SUM[i,PE_FOBO(i)*EXDO(i)]-SUM[i,PWMO(i)*eO*IMO(i)];

** 4.8 Calibration of real (volume) variables computed from price indices
 CTH_REALO(h)    = CTHO(h)/PIXCONO;
 G_REALO         = GO/PIXGVTO;
 GDP_BP_REALO    = GDP_BPO/PIXGDPO;
 GDP_MP_REALO    = GDP_MPO/PIXCONO;
 GFCF_REALO      = GFCFO/PIXINVO;
