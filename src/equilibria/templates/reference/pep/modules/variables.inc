* 5.1 Variable definition (Households + Production subsets)

VARIABLES

**  5.1.1 Volume variables (kept)
 CTH_REAL(h)     Real consumption budget of type h households
 KD(k,j)         Demand for type k capital by industry j
 LD(l,j)         Demand for type l labor by industry j
 VA(j)           Value added of industry j
 CI(j)           Total intermediate consumption of industry j
 XST(j)          Total aggregate output of industry j
 LDC(j)          Industry j demand for composite labor
 KDC(j)          Industry j demand for composite capital
 DI(i,j)         Intermediate consumption of commodity i by industry j
DD(i)           Domestic demand for commodity i
IM(i)           Imports of commodity i

**  5.1.2 Price variables (kept)
 PIXCON          Consumer price index
 R(k,j)          Rental rate of type k capital in industry j
 W(l)            Wage rate of type l labor
 WC(j)           Wage rate of industry j composite labor
 RC(j)           Rental rate of industry j composite capital
 WTI(l,j)        Wage rate paid by industry j for type l labor incl. payroll taxes
 RTI(k,j)        Rental rate paid by industry j for type k capital incl. capital taxes
EXD(i)          Export demand for commodity i
PD(i)           Price of domestic commodity i
PM(i)           Price of imported commodity i
PC(i)           Purchaser price of composite commodity i
PWM(i)          World price of imported commodity i
PWX(i)          World price of exported commodity i
PE(i)           Price received for exported commodity i
PE_FOB(i)       FOB price of exported commodity i

**  5.1.3 Nominal (value) variables (kept)
 CTH(h)          Consumption budget of type h households
 SH(h)           Savings of type h households
 TDH(h)          Income taxes of type h households
 TR(ag,agj)      Transfers from agent agj to agent ag
 YDH(h)          Disposable income of type h households
 YH(h)           Total income of type h households
 YHK(h)          Capital income of type h households
 YHL(h)          Labor income of type h households
 YHTR(h)         Transfer income of type h households

**  5.1.4 Rates and intercepts (kept)
 sh0(h)          Intercept (type h household savings)
 sh1(h)          Slope (type h household savings)
;

* NOTE: Non-essential variables from the full model remain omitted
* for block-by-block testing. We add only what's needed per block.
