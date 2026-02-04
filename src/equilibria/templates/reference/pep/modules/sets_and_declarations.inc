*==============================================================================*
*                        SETS AND DECLARATIONS                                *
*==============================================================================*
*                                                                              *
*           Except where otherwise noted, this work is licensed under          *
*               http://creativecommons.org/licenses/by-nc-sa/3.0/              *
*                                                                              *
*                                                                              *
*  You are free to share, to copy, distribute and transmit the work under      *
*  the following conditions:                                                   *
*                                                                              *
*  - Attribution:         You must attribute the work to:                      *
*                         Veronique Robichaud, Andre Lemelin,                  *
*                         Helene Maisonnave and Bernard Decaluwe.              *
*  - Noncommercial:       You may not use this work for commercial purposes.   *
*  - Share Alike:         If you alter, transform, or build upon this work,    *
*                         you may distribute the resulting work only under     *
*                         the same or similar license to this one.             *
*                                                                              *
*==============================================================================*

* 1 Set definition

** 1.1 Sectors and commodities

SET

J All industries
/
 agr             Agriculture and other primary industries
 ind             Manufacturing and construction
 ser             Services
 adm             Public administration
/

I All commodities
/
 agr             Agriculture and other primary commodities
 food            Food and beverages
 othind          Other manufacturing and construction
 ser             Services
 adm             Public administration
/

I1(I) All commodities except agriculture
/
* agr             Agriculture and other primary commodities
 food            Food and beverages
 othind          Other manufacturing and construction
 ser             Services
 adm             Public administration
/


** 1.2 Production factors

L Labor categories
/
 usk             Unskilled workers
 sk              Skilled workers
/

K Capital categories
/
 cap             Capital
 land            Land
/


** 1.3 Agents

AG All agents
/
 hrp             Poor rural households
 hup             Poor urban households
 hrr             Rich rural households
 hur             Rich urban households
 firm            Firms
 gvt             Government
 row             Rest of the world
/

AGNG(AG) Non governmental agents
/
 hrp             Poor rural households
 hup             Poor urban households
 hrr             Rich rural households
 hur             Rich urban households
 firm            Firms
* gvt             Government
 row             Rest of the world
/

AGD(AG) Domestic agents
/
 hrp             Poor rural households
 hup             Poor urban households
 hrr             Rich rural households
 hur             Rich urban households
 firm            Firms
 gvt             Government
* row             Rest of the world
/

H(AG) Households
/
 hrp             Poor rural households
 hup             Poor urban households
 hrr             Rich rural households
 hur             Rich urban households
/

F(AG) Firms
/
 firm            Firms
/
;

ALIAS (j,jj)
ALIAS (i,ij)
ALIAS (ag,agj)
ALIAS (h,hj)
ALIAS (l,lj)
ALIAS (k,kj)
;
