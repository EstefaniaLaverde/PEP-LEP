#include <iostream>
#include "Data.h"




//global matrices
dmat_type bitsCharCF; // for Byte representation
dynamic_mat_short bits; // for bit representation
dynamic_mat_short matrix; // for reading
dynamic_mat_short matrixH; // for reading
int test = -1;

// allocating memory for a dynamic row of unsigned long long int elements
dynamic_row_short drow_short_new(int size) {
	dynamic_row_short r;
	int size2 = size + 1;
	r = (unsigned long long int*)malloc((size2) * sizeof(unsigned long long int));
	return &(r[0]);
}

//free the memory for a dynamic row
void drow_short_free(dynamic_row_short r) {
	free(&(r[0]));
	r = NULL;
}


// allocating memory for dynamic generator matrix (unsigned long long int)
dynamic_mat_short dmat_short_newh(dynamic_mat_short& c, int k, int n, int q) {
	int i;
	c.n = n;
	c.k = k;
	c.nreal = n;
	c.kreal = k;
	c.q = q;
	int m1 = k + 1;
	c.a = (dynamic_row_short*)malloc((m1) * sizeof(dynamic_row_short));
	//if (c.a == NULL) { if (MM(RD)) c.a = (drowlu*)malloc((m1) * sizeof(drowlu)); }

	ERRORQ(c.a == NULL);
	for (i = 0; i < m1; i++) {
		c.a[i] = drow_short_new(n);
	}

	c.mem = 1;
	return c;
}

//free the memory of dynamic matrix
void dmat_short_free(dynamic_mat_short& c) {
	int i;

	if (c.mem != 0) {
		for (i = 0; i < c.kreal + 1; i++) {
			drow_short_free(c.a[i]);
		}
	}


	free(c.a);
	c.mem = 0;

	return;
}


// main function for allocating dynamic matrix: allocate new memory only if there is need
dynamic_mat_short dmat_short_new(dynamic_mat_short& c, int k, int n, int q) {
	c.k = k;
	c.n = n;

	if (1 == c.mem)
	{
		if ((c.nreal < n) || (c.kreal < k))
		{
			//WORNQ(c.nreal < n);
			//WORNQ(c.mreal < m);
			dmat_short_free(c);
			c = dmat_short_newh(c, k, n, q);
		}
	};
	if (0 == c.mem)
	{
		c.mem = 1;
		c = dmat_short_newh(c, k, n, q);
	}
	return c;
}


// allocating memory for a dynamic row of char elements
drow drow_new(int size) {
	drow r;
	int i = size + 2;
	int j = sizeof(unsigned char);
	int ij = i * j;
	r = (unsigned char*)malloc(ij);

	//if (r == NULL) { if (MM(RD)) r = (unsigned char*)malloc(ij); }

	ERRORQ(r == NULL);


	return &(r[0]);
}

// free memory for dynamic row (char)
void drow_free(drow& r) {
	if (NULL != r)
	{
		free(&(r[0]));
		ERRORQ(NULL == &(r[-1]));
		r = NULL;
	}
}

// allocating memory for dynamic generator matrix (char)
dmat_type dmat_newh(dmat_type& c, int k, int n, int q) {
	//dmat_type c;
	int i;
	c.n = n;
	c.k = k;
	c.nreal = n;
	c.kreal = k;
	c.q = q;
	int m1 = k + 1;
	c.a = (drow*)malloc((m1) * sizeof(drow));
	//if (c.a == NULL) { if (MM(RD)) c.a = (drow*)malloc((m1) * sizeof(drow)); }
	ERRORQ(c.a == NULL);
	for (i = 0; i < k + 1; i++) {
		c.a[i] = drow_new(n);
	}
	c.mem = 1;
	return c;
}

//main function for allocating matrix (char)
dmat_type dmat_new(dmat_type& c, int k, int n, int q) {
	if (!((1 == c.mem) || (0 == c.mem))) {}
		//ERRORQ(!((1 == c.mem) || (0 == c.mem)))
		c.k = k;
	c.q = q;
	c.n = n;
	if (1 == c.mem)
	{
		if ((c.nreal < n) || (c.kreal < k))
		{
			//WORNQ(c.nreal < n);
			//WORNQ(c.mreal < m);
			dmat_free(c);
			c = dmat_newh(c, k, n, q);
		}
	};
	if (0 == c.mem)
	{
		c.mem = 1;
		c = dmat_newh(c, k, n, q);
	}
	return c;
}

// free the memory of dynamic matrix (char)
void dmat_free(dmat_type& c) {
	int i;
	if (!((1 == c.mem) || (0 == c.mem)))
		ERRORQ(!((1 == c.mem) || (0 == c.mem)))

		if (0 == c.mem)
		{
			//WORNQ(0 == c.mem);
			return;
		}

	for (i = 0; i < c.kreal + 1; i++) {
		drow_free(c.a[i]);
	}

	free(c.a);
	c.mem = 0;
	ERRORQ(NULL == c.a);

	return;
}
dmat_type make_dmat(int m, int n, int q) {
	dmat_type c;
	c.mem = 0;
	dmat_new(c, m, n, q);
	return c;

}

//-----------------Data Management functions -----------------------------//#include <iostream>
#include <cstdio>
#include "DataManagement.h"



const unsigned long long int one = 1;


void num_to_coef_gf2(int& k, int& n, int m) {//kratno na 512
	//printf("generator matrix to bit representation (GF2)\n");
	int c = ((n - 1) / 64) + 1;
	for (int row = 0; row < k; row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 <= n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					if (matrix.a[row + 1][el * 64 + shift] == 1) {
						bits.a[row][el] = bits.a[row][el] | (one << (63 - shift));
					}
				}
			}
		}


	}
}

void num_to_coef_gf4(int& k, int& n, int m) {
	int c = (((n - 1) / 64) + 1);
	for (int row = 0; row < 2 * k; row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 <= n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					if (matrix.a[row + 1][el * 64 + shift] == 1) {
						bits.a[row][el] = bits.a[row][el] | (one << (63 - shift));
					}
					else if (matrix.a[row + 1][el * 64 + shift] == 2) {
						bits.a[row][el + c] = bits.a[row][el + c] | (one << (63 - shift));
					}
					else if (matrix.a[row + 1][el * 64 + shift] == 3) {
						bits.a[row][el + c] = bits.a[row][el + c] | (one << (63 - shift));
						bits.a[row][el] = bits.a[row][el] | (one << (63 - shift));
					}
				}
			}
		}
		//cout << bitset<64>(bits[row][0]) << "  " << bitset<64>(bits[row][1]) << endl;
	}
}

void num_to_coef_gf8(int& k, int& n, int m) {
	int c = (((n - 1) / 64) + 1);
	for (int row = 0; row < 3 * k; row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 <= n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					int t = (int) matrix.a[row + 1][el * 64 + shift];
					int ost = t % 2; // ^0
					if (ost) { bits.a[row][el] = bits.a[row][el] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^1
					if (ost) { bits.a[row][el + 1 * c] = bits.a[row][el + 1 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^2
					if (ost) { bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | (one << (63 - shift)); }

				}
			}
		}
	}
}

void num_to_coef_gf16(int& k, int& n, int m) {
	int c = (((n - 1) / 64) + 1);
	for (int row = 0; row < (4 * k); row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 <= n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					int t = (int) matrix.a[row + 1][el * 64 + shift];
					int ost = t % 2; //power 0
					if (ost) { bits.a[row][el] = bits.a[row][el] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; //power 1
					if (ost) { bits.a[row][el + 1 * c] = bits.a[row][el + 1 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; //power 2
					if (ost) { bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; //power 3
					if (ost) { bits.a[row][el + 3 * c] = bits.a[row][el + 3 * c] | (one << (63 - shift)); }


				}
			}
		}
	}
}

void num_to_coef_char2(int& k, int& n, int m) {
	int totalRows = m * k;
	char oneChar = 1;
	for (int row = 0; row < totalRows; row++) {
		for (int col = 0; col < n; col++) {
			bitsCharCF.a[row][col] = matrix.a[row + 1][col];
		}
	}
}




void num_to_coef_char25(int& k, int& n, int m) {
	//printf("generator matrix to coef (char)(GF25)\n");
	int totalRows = m * k;
	char oneChar = 1;
	for (int row = 0; row < totalRows; row++) {
		for (int col = 0; col < n; col++) {
			int temp = (int) matrix.a[row + 1][col];
			bitsCharCF.a[row][col] = temp % 5; // ^0
			temp = temp / 5;
			bitsCharCF.a[row][col + n] = temp % 5; // ^1
		}
	}
}


void num_to_coef_char49(int& k, int& n, int m) {
	int totalRows = m * k;
	char oneChar = 1;
	for (int row = 0; row < totalRows; row++) {
		for (int col = 0; col < n; col++) {
			//bits[row][col] = matrix[row + 1][col];
			int temp = (int) matrix.a[row + 1][col];
			bitsCharCF.a[row][col] = temp % 7; // ^0
			temp = temp / 7;
			bitsCharCF.a[row][col + n] = temp % 7; // ^1
		}
	}
}




void num_to_coef_gf32(int& k, int& n, int m) {
	int c = (((n - 1) / 64) + 1);
	for (int row = 0; row < (5 * k); row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 <= n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					int t = (int) matrix.a[row + 1][el * 64 + shift];
					int ost = t % 2; // ^0
					if (ost) { bits.a[row][el] = bits.a[row][el] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^1
					if (ost) { bits.a[row][el + 1 * c] = bits.a[row][el + 1 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^2
					if (ost) { bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^3
					if (ost) { bits.a[row][el + 3 * c] = bits.a[row][el + 3 * c] | (one << (63 - shift)); }


					t = t / 2;
					ost = t % 2; // ^4
					if (ost) { bits.a[row][el + 4 * c] = bits.a[row][el + 4 * c] | (one << (63 - shift)); }


				}
			}
		}
	}
}



void num_to_coef_gf64(int& k, int& n, int m) {
	int c = (((n - 1) / 64) + 1);
	for (int row = 0; row < (6 * k); row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 <= n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					int t = (int) matrix.a[row + 1][el * 64 + shift];
					int ost = t % 2; // ^0
					if (ost) { bits.a[row][el] = bits.a[row][el] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^1
					if (ost) { bits.a[row][el + 1 * c] = bits.a[row][el + 1 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^2
					if (ost) { bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^3
					if (ost) { bits.a[row][el + 3 * c] = bits.a[row][el + 3 * c] | (one << (63 - shift)); }


					t = t / 2;
					ost = t % 2; // ^4
					if (ost) { bits.a[row][el + 4 * c] = bits.a[row][el + 4 * c] | (one << (63 - shift)); }

					t = t / 2;
					ost = t % 2; // ^5
					if (ost) { bits.a[row][el + 5 * c] = bits.a[row][el + 5 * c] | (one << (63 - shift)); }

				}
			}
		}
	}
}

void num_to_coef_gf27(int k, int n, int m) {
	//printf("generator matrix to bit representation (GF27)\n");
	int c = (((n - 1) / 64) + 1);
	unsigned long long int m_element = 0;
	for (int row = 0; row < 3 * k; row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 < n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					unsigned long long int  n = 0;
					unsigned long long int  t = 0;
					unsigned long long int ost = 0;

					//^0
					m_element = matrix.a[row + 1][el * 64 + shift];
					t = m_element / 3;
					ost = m_element - t * 3;
					if (ost == 0) {
						bits.a[row][el] = bits.a[row][el] | (one << (63 - shift));
						bits.a[row][el + 4 * c] = bits.a[row][el + 4 * c] | (one << (63 - shift));
					}
					else {
						unsigned long long two = (2 & ost) >> 1;
						bits.a[row][el + 4 * c] = bits.a[row][el + 4 * c] | ((one & ost) << (63 - shift));
						bits.a[row][el] = bits.a[row][el] | ((two) << (63 - shift));
					}

					// ^1
					m_element = t;
					t = m_element / 3;
					ost = m_element - t * 3;
					if (ost == 0) {
						bits.a[row][el + c] = bits.a[row][el + c] | (one << (63 - shift));
						bits.a[row][el + 5 * c] = bits.a[row][el + 5 * c] | (one << (63 - shift));
					}
					else {
						unsigned long long two = (2 & ost) >> 1;
						bits.a[row][el + 5 * c] = bits.a[row][el + 5 * c] | ((one & ost) << (63 - shift));
						bits.a[row][el + c] = bits.a[row][el + c] | ((two) << (63 - shift));
					}


					// ^2
					m_element = t;
					t = m_element / 3;
					ost = m_element - t * 3;
					if (ost == 0) {
						bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | (one << (63 - shift));
						bits.a[row][el + 6 * c] = bits.a[row][el + 6 * c] | (one << (63 - shift));
					}
					else {
						unsigned long long two = (2 & ost) >> 1;
						bits.a[row][el + 6 * c] = bits.a[row][el + 6 * c] | ((one & ost) << (63 - shift));
						bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | ((two) << (63 - shift));
					}
				}

			}
		}
	}

}


/* za 512:
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 ||| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 ||| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 ||| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|1 element 1 bit zapisani w 8x64|||2 element 1 bit zapisani w 8x64|||1 element 2 bit zapisani w 8x64|||2 element 2 bit zapisani w 8x64|

*/

void num_to_coef_gf9(int k, int n, int m) {
	int c = (((n - 1) / 64) + 1);// c = 8 za 512;
	unsigned long long int m_element = 0;
	for (int row = 0; row < 2 * k; row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 < n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					unsigned long long int  n = 0;
					unsigned long long int  t = 0;
					unsigned long long int ost = 0;
					m_element = matrix.a[row + 1][el * 64 + shift];
					// ^0
					t = m_element / 3;
					ost = m_element - t * 3;
					if (ost == 0) {
						bits.a[row][el] = bits.a[row][el] | (one << (63 - shift));
						bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | (one << (63 - shift)); // za 9 ->element + 16
					}
					else {
						unsigned long long two = (2 & ost) >> 1;
						bits.a[row][el + 2 * c] = bits.a[row][el + 2 * c] | ((one & ost) << (63 - shift));
						bits.a[row][el] = bits.a[row][el] | ((two) << (63 - shift));
					}

					// ^1
					m_element = t;
					t = m_element / 3;
					ost = m_element - t * 3;
					if (ost == 0) {
						bits.a[row][el + c] = bits.a[row][el + c] | (one << (63 - shift));
						bits.a[row][el + 3 * c] = bits.a[row][el + 3 * c] | (one << (63 - shift));
					}
					else {
						unsigned long long two = (2 & ost) >> 1;
						bits.a[row][el + 3 * c] = bits.a[row][el + 3 * c] | ((one & ost) << (63 - shift));
						bits.a[row][el + c] = bits.a[row][el + c] | ((two) << (63 - shift));
					}
				}

			}
		}
	}

}

void num_to_coef_gf3(int k, int n, int m) {
	int c = (((n - 1) / 64) + 1);
	int m_element = 0;
	for (int row = 0; row < k; row++) {
		for (int el = 0; el < c; el++) {
			if ((el) * 64 < n) {
				for (int shift = 0; shift < 64; shift++) {
					if (el * 64 + shift > (n - 1)) { break; }
					unsigned long long int  n = 0;
					unsigned long long int  t = 0;
					unsigned long long int ost = 0;
					m_element = (int) matrix.a[row + 1][el * 64 + shift];
					t = m_element / 3;
					ost = m_element - t * 3;

					// element of the field GF3 is represented in 2 bits

					if (ost == 0) {
						bits.a[row][el] = bits.a[row][el] | (one << (63 - shift));
						bits.a[row][el + c] = bits.a[row][el + c] | (one << (63 - shift));
					}
					else {
						unsigned long long two = (2 & ost) >> 1;
						bits.a[row][el + c] = bits.a[row][el + c] | ((one & ost) << (63 - shift));
						bits.a[row][el] = bits.a[row][el] | ((two) << (63 - shift));
					}
				}

			}

		}
	}
}
