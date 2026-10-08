#include <iostream>
#include "Polynomials.h"



// multiplication tables for GF(q), q<=64
const int numberOfElements = 65;
int MULH[numberOfElements][numberOfElements], ADDH[numberOfElements][numberOfElements], INVH[numberOfElements], NEGH[numberOfElements];
int ADD[numberOfElements][numberOfElements], MUL[numberOfElements][numberOfElements], INV[numberOfElements], NEG[numberOfElements], DEG[numberOfElements];
int CHIA[numberOfElements];
int CHI = 0;
comp_elements elements[numberOfElements];

char INVHers(char a) {
	return INVH[a];
}
//polinom gf9_irr;
//polmf1.


unsigned long long stepl(int a, int b) //a^b
{
	int i;
	unsigned long long int c;
	c = 1;
	for (i = 1; i <= b; i++)
		c = c * a;
	return c;
}

// decimal representation of a polinomial
int politoint(polinom& a, int q) {
	int tt = 0;
	for (int i = 0; i <= a.grad; i++) {
		if (a.coef[i] > 0) {
			tt = tt + (int) stepl(q, i) * a.coef[i];
		}
	}
	return tt;
}

void copypoli(polinom& a, polinom& b) {
	//incopy++;
	//if (7 == incopy) {
	//	int yy = 0;
	//}
	a.grad = b.grad;
	a.q = b.q;
	for (int i = 0; i <= a.grad; i++) {
		a.coef[i] = b.coef[i];
	}
}

int Russ1(int a, int b, int c) {
	//(* output : RUSS1 = (a + b * c) mod p                *)
	int r = ADDH[a][MULH[b][c]];
	if (r > 1) {
		int jj = 0;
	}
	return r;
}




int multipl_to_dec(int power) {
	for (int i = 0; i < numberOfElements; i++) {
		if (elements[i].power_of_prim == power)
			return elements[i].decimal;
	}
	printf("Did not find elements x^%d\n\n", power);
	return -1;
}

int dec_to_multipl(int decimal) {
	for (int i = 0; i < numberOfElements; i++) {
		if (elements[i].decimal == decimal)
			return elements[i].power_of_prim;
	}
	printf("Did not find elements decimal representation %d\n\n", decimal);
	return -1;
}

int polynomial_to_multipl(polinom& p) {
	int t = politoint(p, p.q);
	return dec_to_multipl(t);
}
void POLREST(polinom& a, polinom& b) {
	//(* input  :  a,b - polynomials over GF(p)      *)
	//(* output :    a = a mod b              *)
	//printpoli(a);
	//printpoli(b);
	int   gq, c, aq, i; //c??
	if (a.grad >= b.grad) {
		gq = a.grad - b.grad + 1;
		if (1 != b.coef[b.grad]) {
			c = INVHers(b.coef[b.grad]);
		}
		else { c = 1; }
		// begin
		while (gq >= 1) {
			//       begin
			gq = gq - 1;
			if ((1 == c) || (0 == a.coef[a.grad])) {
				aq = a.coef[a.grad];
			}
			else { aq = Russ1(0, c, a.coef[a.grad]); }//(* output : RUSS1 = (a + b * c) mod p                *)
			if (0 != aq) { //?
				//            begin
				aq = NEGH[aq];
				for (i = 1; i <= b.grad; i++) {
					a.coef[a.grad - i] = Russ1(a.coef[a.grad - i], aq, b.coef[b.grad - i]);
					//       end;
				}
				a.grad = a.grad - 1;
			}//end;
			if (a.grad >= 0) {// then
			//begin
				while ((a.grad >= 1) && (0 == a.coef[a.grad])) {// do
					a.grad = a.grad - 1;
				}
				if (0 == a.coef[a.grad]) {
					a.grad = -1;
				}
			}//end
		}//end
	}//end
}//end(*Polrest*);


void Poladd1(polinom& f, polinom& g) {
	//(* input : a prime p and polynomials f and g over GF(p).*)
	//(* output :   f = f + g             *)
	int    i, gr;

	if (-1 == f.grad) {
		copypoli(f, g);
		return;
	}
	if (-1 == g.grad) {
		return;
	}
	//inadd++;
	if (f.grad > g.grad) { gr = g.grad; }
	else {

		gr = f.grad;
		f.grad = g.grad;
		for (i = gr + 1; i <= f.grad; i++) {
			f.coef[i] = g.coef[i];
		}
	}//end;
	for (i = 0; i <= gr; i++) {
		f.coef[i] = ADDH[f.coef[i]][g.coef[i]];
	}
	if ((f.grad == gr) && (f.grad > -1)) {

		while ((0 == f.coef[f.grad]) && (f.grad > 0)) {
			f.grad = f.grad - 1;
		}
		if (0 == f.coef[f.grad]) { f.grad = -1; }
	}

}//(*PolADD1*);
void polmf1(polinom& g, polinom& h, polinom& f) {
	//(* output : g = (g*h) mod f  *)

	int   i, j;
	polinom  ghmodf;
	//begin
	if (g.grad >= f.grad) { POLREST(g, f); }
	if (h.grad >= f.grad) { POLREST(h, f); }
	ghmodf.grad = -1;
	if ((g.grad >= 0) && (h.grad >= 0)) {
		ghmodf.grad = g.grad;
		for (i = 0; i <= g.grad; i++) {
			if (0 == g.coef[i])(ghmodf.coef[i] = 0);
			else { ghmodf.coef[i] = Russ1(0, g.coef[i], h.coef[h.grad]); }
		}
		for (j = h.grad - 1; j >= 0; j = j - 1) {
			//(* ghmodf:= ghmodf*x + g*h.coef[j] mod f *)
		   //begin
			ghmodf.grad = ghmodf.grad + 1;
			for (i = ghmodf.grad; i >= 1; i = i - 1) {
				ghmodf.coef[i] = ghmodf.coef[i - 1];
			}
			ghmodf.coef[0] = 0;
			if (ghmodf.grad == f.grad) { POLREST(ghmodf, f); }
			if (0 != h.coef[j]) {
				if (ghmodf.grad <= g.grad) {
					for (i = 0; i <= ghmodf.grad; i++) {
						ghmodf.coef[i] = Russ1(ghmodf.coef[i], g.coef[i], h.coef[j]);
					}
					for (i = ghmodf.grad + 1; i <= g.grad; i++) {
						ghmodf.coef[i] = Russ1(0, g.coef[i], h.coef[j]);
					}
					ghmodf.grad = g.grad;
					while ((ghmodf.grad >= 1) && (0 == ghmodf.coef[ghmodf.grad])) {
						ghmodf.grad = ghmodf.grad - 1;
					}
					if (0 == ghmodf.coef[ghmodf.grad]) {
						ghmodf.grad = -1;
						//end
					}//(ghmodf.grad <= g.grad)
				}
				else {
					for (i = 0; i <= g.grad; i++) {
						ghmodf.coef[i] = Russ1(ghmodf.coef[i], g.coef[i], h.coef[j]);
					}
				}
			}//(0 != h.coef[j])
		}//end
	}// end
//end;


	g.grad = ghmodf.grad;
	for (i = 0; i <= g.grad; i++) {
		g.coef[i] = ghmodf.coef[i];
	}
	//g = ghmodf; //BAGNO DALI PRISVOIAVA MASIVI ne viarvam
}// end(*POLMF1*);

/*
void printpoli(polinom& a) {
	printf("\n st=%d ", a.grad);
	for (int i = 0; i <= a.grad; i++) {
		printf(" %d ", a.coef[i]);
	}
	printf("  -- %d   \n", politoint(a, a.q));
}*/


void maketablecomp(int q) {
	//generating operation tables for composite field
	polinom x, b, a, pa[65], ph;
	//a -> nerazlovim polinom
	//x -> primitiven element
	//b -> edinichen elemet; natrupwa stepenite na primitiwniq element

	pa[0].q = CHI;
	pa[0].grad = -1;
	x.grad = 1;
	x.q = CHI;
	x.coef[0] = 0;
	x.coef[1] = 1;

	a.q = CHI;

	//genearating irreducible polinomials for q=p^m
	switch (q) {
	//definition of primitive polinomials
	case 4:
		a.grad = 2;
		a.coef[0] = 1;
		a.coef[1] = 1;
		a.coef[2] = 1;
		a.coef[3] = 0;
		a.coef[4] = 0;
		break;

	case 8:
		a.grad = 3;
		a.coef[0] = 1;
		a.coef[1] = 1;
		a.coef[2] = 0;
		a.coef[3] = 1;
		a.coef[4] = 0;
		break;

	case 16:
		a.grad = 4;
		a.coef[0] = 1;
		a.coef[1] = 1;
		a.coef[2] = 0;
		a.coef[3] = 0;
		a.coef[4] = 1;
		break;
	case 32:
		a.grad = 5;
		a.coef[0] = 1;
		a.coef[1] = 0;
		a.coef[2] = 1;
		a.coef[3] = 0;
		a.coef[4] = 0;
		a.coef[5] = 1;
		break;
	case 64:
		a.grad = 6;
		a.coef[0] = 1;
		a.coef[1] = 1;
		a.coef[2] = 0;
		a.coef[3] = 0;
		a.coef[4] = 0;
		a.coef[5] = 0;
		a.coef[6] = 1;
		break;
	case 27:
		a.grad = 3;
		a.coef[0] = 1;
		a.coef[1] = 2;
		a.coef[2] = 0;
		a.coef[3] = 1;
		break;
	case 9:
		a.grad = 2;
		a.coef[0] = 2;
		a.coef[1] = 1;
		a.coef[2] = 1;
		break;
	case 25:
		a.grad = 2;
		a.coef[0] = 2;
		a.coef[1] = 1;
		a.coef[2] = 1;
		break;
	case 49:
		a.grad = 2;
		a.coef[0] = 3;
		a.coef[1] = 1;
		a.coef[2] = 1;
		break;

	}
	// K:=2^3;
	//R: = RandomLinearCode(GF(5), 34, 2);
	// time WeightDistribution(R);

		 //K: = GF(3); Primitivni polinomi v MAGMA
		 //P<x>: = PolynomialRing(K);
		 //PrimePolynomials(P, 3);

	b.coef[0] = 1;
	b.coef[1] = 0;
	b.grad = 0;
	b.q = CHI;

	copypoli(pa[1], b);
	//printpoli(b);
	//maketablestat(CHI);

/*
	for (int i = 0; i < CHI; i++) {
		INVH[i] = INV[i];
		NEGH[i] = NEG[i];
	}
	*/

	/*for (int i = 0; i < CHI; i++) {
		for (int j = 0; j < CHI; j++) {
			ADDH[i][j] = ADD[i][j];
			MULH[i][j] = MUL[i][j];
		}
	}*/

	int ord = (int) stepl(a.q, a.grad) - 1;
	DEG[0] = ord + 1;
	DEG[1] = 0;

	elements[0].power_of_prim = ord+1;
	elements[0].decimal = 0;

	elements[1].power_of_prim = 0;
	elements[1].decimal = 1;

	for (int i = 0; i < 8; i++) {
		elements[0].polynomial.coef[i] = 0;
		elements[1].polynomial.coef[i] = 0;
	}

	elements[0].polynomial.grad = 0;
	elements[1].polynomial.grad = 0;

	elements[0].polynomial.q = CHI;
	elements[1].polynomial.q = CHI;


	elements[1].polynomial.coef[0] = 1;

	for (int i = 1; i < ord; i++) {
		polmf1(b, x, a);

		b.q = CHI;

		int temp = politoint(b, CHI);
		copypoli(pa[temp], b);
		DEG[temp] = i;

		copypoli(elements[temp].polynomial,b);
		elements[temp].power_of_prim = i;
		elements[temp].decimal = temp;

		if (1 == temp) {
			int y = 0;
		}
		//printpoli(b);

	}
	polmf1(b, x, a);
	if ((0 != b.grad) && (1 != b.coef[0])) {
		printf("ERROR1");
		int r, error = 0;
		error = scanf("%d", &r);
	}

	for (int i = 0; i <= ord; i++) {
		for (int j = i; j <= ord; j++) {
			copypoli(ph, pa[i]);

			polmf1(ph, pa[j], a);
			int t = politoint(ph, ph.q);
			MUL[i][j] = t;
			MUL[j][i] = t;

			copypoli(ph, pa[i]);

			Poladd1(ph, pa[j]);
			t = politoint(ph, ph.q);
			ADD[i][j] = t;
			ADD[j][i] = t;
		}
	}
	//====================neg==============
	NEG[0] = 0;
	/*for (int ttt = 0; ttt < 27; ttt++) {
		for (int jjj = 0; jjj < 27; jjj++) {
			cout << MUL[ttt][jjj] << ",";
		}
		cout << endl;
	}*/
	for (int i = 1; i <= ord; i++) {

		copypoli(ph, pa[i]);
		for (int j = 0; j <= ph.grad; j++) {
			ph.coef[j] = NEGH[ph.coef[j]];
		}
		int t = politoint(ph, ph.q);
		NEG[i] = t;
	}

	INV[0] = 0;
	INV[1] = 1;
	for (int i = 2; i <= ord; i++) {
		int t = ord - DEG[i];
		copypoli(ph, pa[i]);
		if (37 == i) {
			int y = 0;
		}
		for (int j = 0; j <= ord; j++) {
			if (t == DEG[j]) {
				INV[i] = j;
				break;
			}
		}
	}
	//for (int i = 1; i <= ord; i++) {
	//	DEG[i]++;
	//}
	polinom c, d;
	for (int i = 2; i <= ord; i++) {
		copypoli(c, pa[i]);
		copypoli(d, pa[INV[i]]);
		polmf1(c, d, a);
		if ((0 != c.grad) && (1 != c.coef[0])) {
			printf("ERROR2");
			int r, error =  0;
			error = scanf("%d", &r);
		}
	}
	//1--se predstawa 1,x, x^2,x^3,x^4....x^(q-2)
	//2-- se predstawia x^(q-2),1,x,x^2
	//3=x^4 se predstavia   taka che 1 da e na 4+1 miaso
	//4=x^2 se predstavia taka  che edinicata da e 2+1 miasto
	//end INV=======================================

	// MAT5 e matricata za predstawqne kato dwoichna matrica

	// zanulqvane
	//for (int i = 0; i <= 2 * (q - 1); i++) {
	//	for (int j = 0; j <= 2 * (q - 1); j++) {
	///		MAT5[i][j] = 0;
	//	}
	//}

	//int tt = 0; // zapisvame dwete edinici otpred koitopredstawia 3 kato  dvoichno po stalbove
	//for (int i = 1; i < q; i++) {
	//	tt++;
	//	MAT5[tt][i] = 1;// 1;
	//	tt++;
	//	MAT5[tt][i] = 1;// 1;
	//}
	//tt = 0;
	//slaga edinici po glawnia diagonal na wtorata redica
	 // pres red - na parvia tretia  its
	//for (int i = 1; i < q; i++) {
	//	tt++;
	//	MAT5[tt][(q - 1) + i] = 1;
	//	tt++;
	//}



	//int chih = 1;
	//int i;
	/*while (chih < q) {
		tt = 0;
		for (i = chih; i < (q - 1); i++) {
			tt++;
			MAT5[tt][q + i] = 1;// 1;
			tt++;
			MAT5[tt][q + i] = 1;// 1;
		}
		i = 0;
		while (tt < 2 * (q - 1)) {
			tt++;
			MAT5[tt][q + i] = 1;
			tt++;
			MAT5[tt][q + i] = 1;
			i++;
		}
		chih = chih * CHI;

	}*/



}


void maketable(int q) {
	//generating operation tables
	for (int i = 0; i <= 64; i++) {
		CHIA[i] = 1;
		INV[i] = 0;
		NEG[i] = 0;
		INVH[i] = 0;
		NEGH[i] = 0;
		DEG[i] = 0;
		for(int j = 0; j <= 64; j++) {
			MULH[i][j] = 0;
			MUL[i][j] = 0;
			ADD[i][j] = 0;
			ADDH[i][j] = 0;
		}
	}
	//characteristic of the fields
	CHIA[2] = 2;
	CHIA[3] = 3;
	CHIA[4] = 2;
	CHIA[5] = 5;
	CHIA[7] = 7;
	CHIA[8] = 2;
	CHIA[9] = 3;
	CHIA[11] = 11;
	CHIA[13] = 13;
	CHIA[16] = 2;
	CHIA[17] = 17;
	CHIA[19] = 19;
	CHIA[23] = 23;
	CHIA[25] = 5;
	CHIA[27] = 3;
	CHIA[29] = 29;
	CHIA[31] = 31;
	CHIA[32] = 2;
	CHIA[37] = 37;
	CHIA[41] = 41;
	CHIA[43] = 43;
	CHIA[47] = 47;
	CHIA[49] = 7;
	CHIA[53] = 53;
	CHIA[59] = 59;
	CHIA[61] = 61;
	CHIA[64] = 2;

	CHI = CHIA[q];
	if (CHI == q) { //!=
		for (int i = 0; i < q; i++) {
			for (int j = i; j < q; j++) {
				ADD[i][j] = (i + j) % q;
				ADD[j][i] = ADD[i][j];
				MUL[i][j] = (i * j) % q;
				MUL[j][i] = MUL[i][j];
			}
		}

		//maketablecomp(q);
		//return;
	//}

		for (int i = 1; i < q; i++) {
			for (int j = 1; j < q; j++) {
				if (1 == MUL[i][j]) {
					INV[i] = j;
				}
				if (0 == ADD[i][j]) {
					NEG[i] = j;
				}
			}
		}
	}
	else { // for composite fields

		for (int i = 0; i < CHI; i++) {
			for (int j = i; j < CHI; j++) {
				ADDH[i][j] = (i + j) % CHI;
				ADDH[j][i] = ADDH[i][j];
				MULH[i][j] = (i * j) % CHI;
				MULH[j][i] = MULH[i][j];
			}
		}

		for (int i = 1; i < CHI; i++) {
			for (int j = 1; j < CHI; j++) {
				if (1 == MULH[i][j]) {
					INVH[i] = j;
				}
				if (0 == ADDH[i][j]) {
					NEGH[i] = j;
				}
			}
		}
		maketablecomp(q);
	}

}


int multiply(int m, int n) {
	return MUL[m][n];
}

void multiplyGF9(int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[3][matrix.a[i][j]];
		}
	}
}

void multiplyGF4( int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[2][matrix.a[i][j]];
		}
	}
}
void multiplyGF8( int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[2][matrix.a[i][j]];
			matrix.a[2 * k + i][j] = MUL[4][matrix.a[i][j]];
		}
	}
}

void multiplyGF16(int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[2][matrix.a[i][j]];
			matrix.a[2 * k + i][j] = MUL[4][matrix.a[i][j]];
			matrix.a[3 * k + i][j] = MUL[8][matrix.a[i][j]];
		}
	}
}

void multiplyGF25(int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[5][matrix.a[i][j]];
		}
	}
}

void multiplyGF49( int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[7][matrix.a[i][j]];
		}
	}
}

void multiplyGF32(int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[2][matrix.a[i][j]];
			matrix.a[2 * k + i][j] = MUL[4][matrix.a[i][j]];
			matrix.a[3 * k + i][j] = MUL[8][matrix.a[i][j]];
			matrix.a[4 * k + i][j] = MUL[16][matrix.a[i][j]];
		}
	}
}


void multiplyGF64(int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[2][matrix.a[i][j]];
			matrix.a[2 * k + i][j] = MUL[4][matrix.a[i][j]];
			matrix.a[3 * k + i][j] = MUL[8][matrix.a[i][j]];
			matrix.a[4 * k + i][j] = MUL[16][matrix.a[i][j]];
			matrix.a[5 * k + i][j] = MUL[32][matrix.a[i][j]];
		}
	}
}

void multiplyGF27(int k, int n) {
	for (int i = 1; i <= k; i++) {
		for (int j = 0; j < n; j++) {
			matrix.a[k + i][j] = MUL[3][matrix.a[i][j]];
			matrix.a[2 * k + i][j] = MUL[9][matrix.a[i][j]];
		}
	}
}

int add_vector( int n, int h, int i, int q1) {
	int sum = 0, teglo = 0;
	for (int j = 0; j < n; j++) {
		sum = ADD[matrixH.a[h - 1][j]][
			MUL[matrix.a[i][j]][q1]
		];
		matrixH.a[h][j] = sum; //// a[h] = a[h - 1] + a[i] * q1
		if (sum != 0) { teglo++; }
	}
	return teglo;
}

