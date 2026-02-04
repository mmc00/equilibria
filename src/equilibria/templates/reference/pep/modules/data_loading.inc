*==============================================================================*
*                              DATA LOADING MODULE                            *
*==============================================================================*
*                                                                              *
*  This module handles all data loading operations for the PEP CGE model      *
*  including SAM data, parameter data, and initial value assignments          *
*                                                                              *
*==============================================================================*

* 3 Data

** 3.1 Data from the SAM
*  SAM data are nominal values. However, several volume variables are
*  provisionally set equal to the corresponding nominal SAM value. Once the
*  benchmark prices have been set or calibrated, volumes will be re-calculated
*  (section 4.4).

PARAMETER
SAM(*,*,*,*);

** 3.1.1 SAM Data Loading Configuration
$onEcho > connect.yaml
- ExcelReader:
    file: SAM-V2_0.xlsx
    symbols:
    - name: SAM
      range: SAM!A1:AJ39
      rowDimension: 2
      columnDimension: 2
- GDXWriter:
    file: SAM-V2_0.gdx
    symbols:
    - name: SAM
$offEcho

** 3.1.2 Execute SAM Data Loading
$call gamsconnect connect.yaml
$GDXIN SAM-V2_0.gdx
$LOAD SAM
$GDXIN

* Manual declaration of data labels to prevent Error 116
SET data_labels / 'I', 'Agents', 'J', 'X', 'OTH', 'INV', 'VSTK', 'gvt', 'ROW', 'K', 'L', 'TAX', 'TI', 'TM', 'GVT', 'firm' /;

** 3.1.3 SAM Data Assignment to Model Variables
* Consumption flows
 CO(i,h)         = SAM('I',i,'Agents',h);
 CGO(i)          = SAM('I',i,'Agents','gvt');

* Production and trade flows
 DSO(j,i)        = SAM('J',j,'I',i);
 DDO(i)          = SUM[j,DSO(j,i)];
 DIO(i,j)        = SAM('I',i,'J',j);
 EXO(j,i)        = SAM('J',j,'X',i);
 EXDO(i)         = SAM('X',i,'Agents','ROW');

* Investment and inventory
 INVO(i)         = SAM('I',i,'OTH','INV');
 VSTKO(i)        = SAM('I',i,'OTH','VSTK');

* Imports
 IMO(i)          = SAM('Agents','ROW','I',i);

* Factor demands
 KDO(k,j)        = SAM('K',k,'J',j);
 LDO(l,j)        = SAM('L',l,'J',j);

* Savings by agent
 SFO(f)          = SAM('OTH','INV','Agents',f);
 SGO             = SAM('OTH','INV','Agents','GVT');
 SHO(h)          = SAM('OTH','INV','Agents',h);
 SROWO           = SAM('OTH','INV','Agents','ROW');

* Tax revenues
 TDFO(f)         = SAM('Agents','TAX','Agents',f);
 TDHO(h)         = SAM('Agents','TAX','Agents',h);
 TICO(i)         = SAM('Agents','TI','I',i);
 TIKO(k,j)       = SAM('Agents',k,'J',j);
 TIMO(i)         = SAM('Agents','TM','I',i);
 TIPO(j)         = SAM('Agents','GVT','J',j);
 TIXO(i)         = SAM('Agents','GVT','X',i);
 TIWO(l,j)       = SAM('Agents',l,'J',j);

* Transfers
 TRO(ag,agj)     = SAM('Agents',ag,'Agents',agj);

* Income distribution parameters
 lambda_RK(ag,k) = SAM('Agents',ag,'K',k);
 lambda_WL(h,l)  = SAM('Agents',h,'L',l);

* Margin rates
 tmrg(i,ij)      = SAM('I',i,'I',ij);
 tmrg_X(i,ij)    = SAM('I',i,'X',ij);


** 3.2 Other data
** Some parameters cannot be calibrated using SAM values
** Exogenous parameters

PARAMETER
PARJ(j,*), PARI(i,*), PARJI(j,i), PARAG(*,*);

** 3.2.1 Parameter Data Loading Configuration
$onEcho > connectpar.yaml
- ExcelReader:
    file: VAL_PAR.xlsx
    symbols:
    - name: PARJ
      range: PAR!A5:E9
    - name: PARI
      range: PAR!A12:C17
    - name: PARJI
      range: PAR!A20:F24
    - name: PARAG
      range: PAR!A27:F37
- GDXWriter:
    file: VAL_PAR.gdx
    symbols:
    - name: PARJ
    - name: PARI
    - name: PARJI
    - name: PARAG
$offEcho

** 3.2.2 Execute Parameter Data Loading
$call gamsconnect connectpar.yaml
$GDXIN VAL_PAR.gdx
$LOAD PARJ, PARI, PARJI, PARAG
$GDXIN

** 3.2.3 Parameter Assignment from External Data

*** CES and CET elasticities
 sigma_KD(j)     = PARJ(j,'sigma_KD');
 sigma_LD(j)     = PARJ(j,'sigma_LD');
 sigma_M(i)      = PARI(i,'sigma_M');
 sigma_VA(j)     = PARJ(j,'sigma_VA');
 sigma_X(j,i)    = PARJI(j,i);
 sigma_XT(j)     = PARJ(j,'sigma_XT');

*** Elasticity of international demand for exported commodity i
 sigma_XD(i)     = PARI(i,'sigma_XD');

*** LES parameters
 frisch(h)       = PARAG('frisch',h);
 sigma_y(i,h)    = PARAG(i,h);

*** Intercepts of transfers, direct taxes and savings
*  One can either choose to assign a value to the intercept and calibrate
*  the slopes accordingly, or the other way around. This type of modelling
*  can be useful to take into account known marginal savings or taxation rates
*  or to deal with negative average saving rates in cases where savings are
*  negative for some household groups.
*  When no further information is available, one can simply set the intercepts
*  to zero and calibrate an average rate: this is what we do here.
 sh0O(h)         = PARAG('sh0O',h);
 tr0O(h)         = PARAG('tr0O',h);
 ttdf0O(f)       = PARAG('ttdf0O',f);
 ttdh0O(h)       = PARAG('ttdh0O',h);

** 3.3 Initial Value Assignments

*** Price elasticity (should be set equal to one when verifying model homogeneity)
 eta             = 1;

*** Initial price normalization (all set to 1 for calibration)
 PLO(i)          = 1;
 PEO(i)          = 1;
 eO              = 1;
 PWMO(i)         = 1;
 WO(l)           = 1;
 RKO(k)          = 1;
 RO(k,j)         = RKO(k);
