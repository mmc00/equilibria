* 5.4 Households block: definitions aligned with original PEP equations

EQ10(h)..       YH(h) =e= YHL(h)+YHK(h)+YHTR(h);
EQ11(h)..       YHL(h) =e= SUM{l,lambda_WL(h,l)*W(l)*SUM[j$LDO(l,j),LD(l,j)]};
EQ12(h)..       YHK(h) =e= SUM{k,lambda_RK(h,k)*SUM[j$KDO(k,j),R(k,j)*KD(k,j)]};
EQ13(h)..       YHTR(h) =e= SUM[ag,TR(h,ag)];
EQ14(h)..       YDH(h) =e= YH(h)-TDH(h)-TR('gvt',h);
EQ15(h)..       CTH(h) =e= YDH(h)-SH(h)-SUM[agng,TR(agng,h)];
EQ16(h)..       SH(h) =e= PIXCON**eta*sh0(h)+sh1(h)*YDH(h);
EQ94(h)..       CTH_REAL(h) =e= CTH(h)/PIXCON;