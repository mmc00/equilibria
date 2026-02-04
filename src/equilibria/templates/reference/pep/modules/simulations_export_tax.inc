*==============================================================================*
*                               SIMULATIONS MODULE                              *
*==============================================================================*
* This include centralizes policy simulation scenarios for the PEP CGE model.   *
* Uncomment or edit scenarios here, and include from model_solution.inc.        *
*==============================================================================*

*** 6.4.1 Import price shock
*  25% increase of international import price of agriculture
* PWM.fx('agr')   = PWMO('agr')*1.25;

*** 6.4.2 Export tax reduction (ACTIVE SIMULATION)
*  25% decrease of export tax rates on all commodities
	ttix.fx(i)      = ttixO(i)*0.75;

*** 6.4.3 Government spending increase
*  20% increase of public expenditures
* G.fx            = GO*1.2;

*** 6.4.4 Additional simulation examples (commented out)
*  Income tax increase
* ttdh1.fx(h)     = ttdh1O(h)*1.1;

*  Import tariff reduction
* ttim.fx(i)      = ttimO(i)*0.9;

*  Labor supply increase
* LS.fx(l)        = LSO(l)*1.05;

*  Productivity shock (total factor productivity)
* B_VA(j)         = B_VA(j)*1.03;
