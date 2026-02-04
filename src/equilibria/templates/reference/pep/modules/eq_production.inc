* 5.3.1 Production block (ported from original PEP)

EQ1(j)..        VA(j) =e= v(j)*XST(j);

EQ2(j)..        CI(j) =e= io(j)*XST(j);

EQ3(j)..        VA(j) =e= B_VA(j)*{
						 [beta_VA(j)*LDC(j)**(-rho_VA(j))]$LDCO(j)
						+[(1-beta_VA(j))*KDC(j)**(-rho_VA(j))]$KDCO(j)
							}**(-1/rho_VA(j));

EQ4(j)$[LDCO(j) and KDCO(j)]..
			 LDC(j) =e= {[beta_VA(j)/(1-beta_VA(j))]*[RC(j)/WC(j)]}
					 **sigma_VA(j)*KDC(j);

EQ5(j)$LDCO(j)..
			 LDC(j) =e= B_LD(j)*SUM[l$LDO(l,j),beta_LD(l,j)*LD(l,j)
					  **(-rho_LD(j))]**(-1/rho_LD(j));

EQ6(l,j)$LDO(l,j)..
			 LD(l,j) =e= [beta_LD(l,j)*WC(j)/WTI(l,j)]**sigma_LD(j)
					   *B_LD(j)**(sigma_LD(j)-1)*LDC(j);

EQ7(j)$KDCO(j)..
			 KDC(j) =e= B_KD(j)*SUM[k$KDO(k,j),beta_KD(k,j)*KD(k,j)
					  **(-rho_KD(j))]**(-1/rho_KD(j));

EQ8(k,j)$KDO(k,j)..
			 KD(k,j) =e= [beta_KD(k,j)*RC(j)/RTI(k,j)]**sigma_KD(j)
					   *B_KD(j)**(sigma_KD(j)-1)*KDC(j);

EQ9(i,j)..      DI(i,j) =e= aij(i,j)*CI(j);