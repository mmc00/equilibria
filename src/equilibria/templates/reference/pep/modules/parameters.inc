*==============================================================================*
*                              PARAMETERS MODULE                              *
*==============================================================================*
*                                                                              *
*  This module contains all parameter declarations for the PEP CGE model      *
*  including functional form parameters, benchmark variables, and rates       *
*                                                                              *
*==============================================================================*

* 2 Parameters and benchmark variables definition

** 2.1 Parameters

PARAMETERS
 aij(i,j)          Input output coefficient
 B_KD(j)           Scale parameter (CES - composite capital)
 B_LD(j)           Scale parameter (CES - composite labor)
 B_M(i)            Scale parameter (CES - composite commodity)
 B_VA(j)           Scale parameter (CES - value added)
 B_X(j,i)          Scale parameter (CET - exports and local sales)
 B_XT(j)           Scale parameter (CET - total output)
 beta_KD(k,j)      Share parameter (CES - composite capital)
 beta_LD(l,j)      Share parameter (CES - composite labor)
 beta_M(i)         Share parameter (CES - composite commodity)
 beta_VA(j)        Share parameter (CES - value added)
 beta_X(j,i)       Share parameter (CET - exports and local sales)
 beta_XT(j,i)      Share parameter (CET - total output)
 eta               Price elasticity of indexed transfers and parameters
 frisch(h)         Frisch parameter (LES function)
 gamma_GVT(i)      Share of commodity i in total current public expenditures on goods and services
 gamma_INV(i)      Share of commodity i in total investment expenditures
 gamma_LES(i,h)    Marginal share of commodity i in household h consumption budget
 io(j)             Coefficient (Leontief - intermediate consumption)
 kmob              Flag parameter (1 if capital is mobile)
 lambda_RK(ag,k)   Share of type k capital income received by agent ag
 lambda_TR(ag,agj) Share parameter (transfer functions)
 lambda_WL(h,l)    Share of type l labor income received by type h households
 rho_KD(j)         Elasticity parameter (CES - composite capital)
 rho_LD(j)         Elasticity parameter (CES - composite labor)
 rho_M(i)          Elasticity parameter (CES - composite commodity)
 rho_VA(j)         Elasticity parameter (CES - value added)
 rho_X(j,i)        Elasticity parameter (CET - exports and local sales)
 rho_XT(j)         Elasticity parameter (CET - total output)
 sigma_KD(j)       Elasticity (CES - composite capital)
 sigma_LD(j)       Elasticity (CES - composite labor)
 sigma_M(i)        Elasticity (CES - composite commodity)
 sigma_VA(j)       Elasticity (CES - value added)
 sigma_X(j,i)      Elasticity (CET - exports and local sales)
 sigma_XT(j)       Elasticity (CET - total output)
 sigma_XD(i)       Price elasticity of the world demand for exports of product i
 sigma_Y(i,h)      Income elasticity of consumption
 tmrg(i,ij)        Rate of margin i applied to commodity ij
 tmrg_X(i,ij)      Rate of margin i applied to exported commodity i
 v(j)              Coefficient (Leontief - value added)


** 2.2 Variables - Benchmark
* Benchmark values of variables are parameters.
* Their acronyms are the corresponding variable names,
* followed by the letter "O".

**  2.2.1 Volume variables
 CO(i,h)           Consumption of commodity i by type h households
 CGO(i)            Public consumption of commodity i
 CIO(j)            Total intermediate consumption of industry j
 CMINO(i,h)        Minimum consumption of commodity i by type h households
 CTH_REALO(h)      Real consumption budget of type h households
 DDO(i)            Domestic demand for commodity i produced locally
 DIO(i,j)          Intermediate consumption of commodity i by industry j
 DITO(i)           Total intermediate demand for commodity i
 DSO(j,i)          Supply of commodity i by sector j to the domestic market
 EXO(j,i)          Quantity of product i exported by sector j
 EXDO(i)           World demand for exports of product i
 G_REALO           Real current government expenditures on goods and services
 GDP_BP_REALO      Real GDP at basic prices
 GDP_MP_REALO      Real GDP at market prices
 GFCF_REALO        Real gross fixed capital formation
 IMO(i)            Quantity of product i imported
 INVO(i)           Final demand of commodity i for investment purposes (GFCF)
 KDO(k,j)          Demand for type k capital by industry j
 KDCO(j)           Industry j demand for composite capital
 KSO(k)            Supply of type k capital
 LDO(l,j)          Demand for type l labor by industry j
 LDCO(j)           Industry j demand for composite labor
 LSO(l)            Supply of type l labor
 MRGNO(i)          Demand for commodity i as a trade or transport margin
 QO(i)             Quantity demanded of composite commodity i
 VAO(j)            Value added of industry j
 VSTKO(i)          Inventory change of commodity i
 XSO(j,i)          Industry j production of commodity i
 XSTO(j)           Total aggregate output of industry j

**  2.2.2 Price variables
 eO                Exchange rate (price of foreign currency in local currency)
 PO(j,i)           Basic price of industry j's production of commodity i
 PCO(i)            Purchaser price of composite comodity i (including all taxes and margins)
 PCIO(j)           Intermediate consumption price index of industry j
 PDO(i)            Price of local product i sold on the domestic market (including all taxes and margins)
 PEO(i)            Price received for exported commodity i (excluding export taxes)
 PE_FOBO(i)        FOB price of exported commodity i (in local currency)
 PIXCONO           Consumer price index
 PIXGDPO           GDP deflator
 PIXGVTO           Public expenditures price index
 PIXINVO           Investment price index
 PLO(i)            Price of local product i (excluding all taxes on products)
 PMO(i)            Price of imported product i (including all taxes and tariffs)
 PPO(j)            Industry j unit cost including taxes directly related to the use of capital and labor but excluding other taxes on production
 PTO(j)            Basic price of industry j's output
 PVAO(j)           Price of industry j value added (including taxes on production directly related to the use of capital and labor)
 PWMO(i)           World price of imported product i (expressed in foreign currency)
 PWXO(i)           World price of exported product i (expressed in foreign currency)
 RO(k,j)           Rental rate of type k capital in industry j
 RCO(j)            Rental rate of industry j composite capital
 RKO(k)            Rental rate of type k capital (if capital is mobile)
 RTIO(k,j)         Rental rate paid by industry j for type k capital including capital taxes
 WO(l)             Wage rate of type l labor
 WCO(j)            Wage rate of industry j composite labor
 WTIO(l,j)         Wage rate paid by industry j for type l labor including payroll taxes

**  2.2.3 Nominal (value) variables
 CABO              Current account balance
 CTHO(h)           Consumption budget of type h households
 GO                Current government expenditures on goods and services
 GDP_BPO           GDP at basic prices
 GDP_FDO           GDP at purchasers' prices from the perspective of final demand
 GDP_IBO           GDP at market prices (income-based)
 GDP_MPO           GDP at market prices
 GFCFO             Gross fixed capital formation
 ITO               Total investment expenditures
 SFO(f)            Savings of type f businesses
 SGO               Government savings
 SHO(h)            Savings of type h households
 SROWO             Rest-of-the-world savings
 TDFO(f)           Income taxes of type f businesses
 TDFTO             Total government revenue from business income taxes
 TDHO(h)           Income taxes of type h households
 TDHTO             Total government revenue from household income taxes
 TICO(i)           Government revenue from indirect taxes on product i
 TICTO             Total government receipts of indirect taxes on commodities
 TIKO(k,j)         Government revenue from taxes on type k capital used by industry j
 TIKTO             Total government revenue from from taxes on capital
 TIMO(i)           Government revenue from import duties on product i
 TIMTO             Total government revenue from import duties
 TIPO(j)           Government revenue from taxes on industry j production (excluding taxes directly related to the use of capital and labor)
 TIPTO             Total government revenue from production taxes (excluding taxes directly related to the use of capital and labor)
 TIWO(l,j)         Government revenue from payroll taxes on type l labor in industry j
 TIWTO             Total government revenue from payroll taxes
 TIXO(i)           Government revenue from export taxes on product i
 TIXTO             Total government revenue from export taxes
 TPRCTSO           Total government revenue from taxes on products and imports
 TPRODNO           Total government revenue from other taxes on production
 TRO(ag,agj)       Transfers from agent agj to agent ag
 YDFO(f)           Disposable income of type f businesses
 YDHO(h)           Disposable income of type h households
 YFO(f)            Total income of type f businesses
 YFKO(f)           Capital income of type f businesses
 YFTRO(f)          Transfer income of type f businesses
 YGO               Total government income
 YGKO              Government capital income
 YGTRO             Government transfer income
 YHO(h)            Total income of type h households
 YHKO(h)           Capital income of type h households
 YHLO(h)           Labor income of type h households
 YHTRO(h)          Transfer income of type h households
 YROWO             Rest-of-the-world income

**  2.2.4 Rates and intercepts
 sh0O(h)           Intercept (type h household savings)
 sh1O(h)           Slope (type h household savings)
 tr0O(h)           Intercept (transfers by type h households to government)
 tr1O(h)           Marginal rate of transfers by type h households to government
 ttdf0O(f)         Intercept (income taxes of type f businesses)
 ttdf1O(f)         Marginal income tax rate of type f businesses
 ttdh0O(h)         Intercept (income taxes of type h households)
 ttdh1O(h)         Marginal income tax rate of type h households
 tticO(i)          Tax rate on commodity i
 ttikO(k,j)        Tax rate on type k capital used in industry j
 ttimO(i)          Rate of taxes and duties on imports of commodity i
 ttipO(j)          Tax rate on the production of industry j
 ttiwO(l,j)        Tax rate on type l worker compensation in industry j
 ttixO(i)          Export tax rate on exported commodity i
;
