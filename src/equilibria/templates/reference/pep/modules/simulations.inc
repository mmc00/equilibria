*==============================================================================*
*                               SIMULATIONS MODULE                              *
*==============================================================================*
* This include centralizes policy simulation scenarios for the PEP CGE model.   *
* The scenario is controlled by the --scenario=NAME command line argument.      *
*                                                                               *
* Available scenarios:                                                          *
*   baseline       - No policy changes (default)                                *
*   export-tax     - Reduce export taxes by 25%                                 *
*   gov-spending   - Increase government spending by 20%                        *
*   import-shock   - Increase import prices by 25%                              *
*   agr-shock      - Increase agriculture import price by 25%                   *
*   food-shock     - Increase food import price by 25%                          *
*   othind-shock   - Increase other industry import price by 25%                *
*   ser-shock      - Increase services import price by 25%                      *
*   tax-reform     - Reduce commodity taxes, increase income taxes              *
*                                                                               *
* Usage: gams PEP-1-1_v2_1_modular.gms --scenario=export-tax                    *
*==============================================================================*

* Default to baseline if no scenario specified
$if not set scenario $set scenario baseline

*==============================================================================*
* SCENARIO IMPLEMENTATIONS
*==============================================================================*

$ifthen.scen %scenario% == baseline
* ------------------------------------------------------------------------------
* BASELINE - No policy changes
* ------------------------------------------------------------------------------
*   No policy shocks applied - running baseline calibration

$elseif.scen %scenario% == export-tax
* ------------------------------------------------------------------------------
* EXPORT TAX REDUCTION - 25% decrease in export tax rates
* ------------------------------------------------------------------------------
    ttix.fx(i) = ttixO(i) * 0.75;

$elseif.scen %scenario% == gov-spending
* ------------------------------------------------------------------------------
* GOVERNMENT SPENDING INCREASE - 20% increase in public expenditures
* ------------------------------------------------------------------------------
    G.fx = GO * 1.2;

$elseif.scen %scenario% == import-shock
* ------------------------------------------------------------------------------
* IMPORT PRICE SHOCK - 25% increase in world import prices (ALL sectors)
* ------------------------------------------------------------------------------
    PWM.fx(i) = PWMO(i) * 1.25;

$elseif.scen %scenario% == agr-shock
* ------------------------------------------------------------------------------
* AGRICULTURE IMPORT SHOCK - 25% increase for agriculture only
* ------------------------------------------------------------------------------
    PWM.fx('agr') = PWMO('agr') * 1.25;

$elseif.scen %scenario% == food-shock
* ------------------------------------------------------------------------------
* FOOD IMPORT SHOCK - 25% increase for food only
* ------------------------------------------------------------------------------
    PWM.fx('food') = PWMO('food') * 1.25;

$elseif.scen %scenario% == othind-shock
* ------------------------------------------------------------------------------
* OTHER INDUSTRY IMPORT SHOCK - 25% increase for othind only
* ------------------------------------------------------------------------------
    PWM.fx('othind') = PWMO('othind') * 1.25;

$elseif.scen %scenario% == ser-shock
* ------------------------------------------------------------------------------
* SERVICES IMPORT SHOCK - 25% increase for services only
* ------------------------------------------------------------------------------
    PWM.fx('ser') = PWMO('ser') * 1.25;

$elseif.scen %scenario% == tax-reform
* ------------------------------------------------------------------------------
* TAX REFORM - Reduce commodity taxes, increase income taxes
* ------------------------------------------------------------------------------
    ttix.fx(i) = ttixO(i) * 0.80;      * 20% reduction in export taxes
    ttic.fx(i) = tticO(i) * 0.90;      * 10% reduction in commodity taxes
    ttdh1.fx(h) = ttdh1O(h) * 1.10;    * 10% increase in household income tax

$elseif.scen %scenario% == tax-sales
* ------------------------------------------------------------------------------
* SALES TAX INCREASE - 15% increase in sales tax rates
* ------------------------------------------------------------------------------
    ttic.fx(i) = tticO(i) * 1.15;

$else.scen
* ------------------------------------------------------------------------------
* UNKNOWN SCENARIO - Error
* ------------------------------------------------------------------------------
$   error "Unknown scenario: %scenario%. Available: baseline, export-tax, gov-spending, import-shock, agr-shock, food-shock, othind-shock, ser-shock, tax-reform"

$endif.scen

*==============================================================================*
* END OF SIMULATIONS MODULE
*==============================================================================*
