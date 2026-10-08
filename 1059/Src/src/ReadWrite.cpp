#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <ctime>

#include"ReadWrite.h"

using namespace std;

void write_multpl(int dec,  FILE* write) {
	int m = dec_to_multipl(dec);
	if(m==-1){
		fclose(write);
		FILE* err = fopen("error.txt", "w");
		fprintf(err, "Error in multiplicative representation of the elements!!\n");
		fclose(err);
		return;
	}
	fprintf(write, "y^%d,", m);

}


void write_multpl_magma(int dec, FILE* write) {
	int m = dec_to_multipl(dec);
	if (m == -1) {
		fclose(write);
		FILE* err = fopen("error.txt", "w");
		fprintf(err, "Error in multiplicative representation of the elements!!\n");
		fclose(err);
		return;
	}
	if (dec < CHI) {
		fprintf(write, "%*d ",5, dec);
	}
	else {
		fprintf(write, "%*$.1^%d ",2, m);
	}
	

}



int charStriingToInt(char* number, int len) {
	// for reading decimal numbers >10
	int d = 1, t = 0, res = 0;
	for (int i = len - 1; i >= 0; i--) {
		if (number[i] == '0') t = 0;
		if (number[i] == '1') t = 1;
		if (number[i] == '2') t = 2;
		if (number[i] == '3') t = 3;
		if (number[i] == '4') t = 4;
		if (number[i] == '5') t = 5;
		if (number[i] == '6') t = 6;
		if (number[i] == '7') t = 7;
		if (number[i] == '8') t = 8;
		if (number[i] == '9') t = 9;
		res = res + t * d;
		d = d * 10;
	}
	return res;
}

bool readMatrix(FILE * fileName, int& n, int& k, int& q) {
	//reading generator matrix form open input file
	/*
	char* ch;
	char cc;
	char chh[301];

	//fscanf(fpr, "%d", &k);
	//fscanf(fpr, "%d", &n);

	//fscanf(fpr, "%d", &q);
	
	//fscanf(fpr, "%s", &chh);
	

	fgets(chh, 300, fpr);
	int y = 0;



	// int j=sizeof( char);
	ch = (char*)malloc((n) * 4);
	int lch = n * 4;
	//fgets(ch, sizeof(ch), fpr);
	//dmat_new(genmat, m, n, q); //otdelia pamet za matricata ako ne e otdelena

	//maketable(genmat.q); // zaregda tablicite na poleto

	//genmat.k = genmat.m;
	char c = 1;
	// while (c!='\n'){
	//     c=getc(fpr);}
	for (int kk = 1; kk <= matrix.k; kk++) {
		fgets(ch, lch, fpr);
		int len = 0;
		while (ch[len])
		{
			len++;
		}
		//int y=sizeof(ch*);
		if ('\n' == ch[len - 1]) {
			len--;
		}
		int com = 0;
		for (int i = 0; i < len; i++) {
			if (',' == ch[i])
			{
				com++;
			}
			// if('\n'==ch[i]){len=i-1;break;}
		}

		char ct, st[10];
		int tt = -1, nn = 0, comh = 0;
		if ((com > 0) && (com != n)) {
			return true;
			//ERRORQ((com > 0) && (com != n));
		}
		if (com > 0) { // ako ima zapetajki
			for (int i = 0; i < len; i++) {
				if (',' != ch[i]) {
					if ((48 <= ch[i]) && (48 + q - 1 >= ch[i])) {
						tt++; st[tt] = ch[i];
					}
					else {
						return true;
						//ERRORQ(!((48 <= ch[i]) && (48 + q - 1 >= ch[i])))
					}
				}
				if (',' == ch[i])
				{
					tt++;
					st[tt] = '\n';
					int r = atoi(st);
					nn++;
					matrix.a[kk][nn] = atoi(st);;
					tt = -1;
					comh++;
					if (comh == n) {
						i = len;
						break;
					}
				}

			}
		}
		else {// ako niama zapetajki megdu elementite - bsiako chislo e element
			for (int i = 0; i < n; i++) {
				if ((48 <= ch[i]) && (48 + q - 1 >= ch[i])) {
				}
				else {
					return true;
					//ERRORQ(!((48 <= ch[i]) && (48 + q - 1 >= ch[i])))
				}

				st[0] = ch[i];
				st[1] = '\n';
				int r = atoi(st);
				nn++;
				matrix.a[kk][nn] = atoi(st);;
			}
		}// kraj ako niama zapetajki
	}
	free(ch);
	return false;
	*/
	bool err = false;
	if (fileName == NULL) {
		//if there is an error in opening the file - > write error in error.txt
		FILE* errf = fopen("error.txt", "w");
		fprintf(errf, "cannot open File!!\n");
		fclose(errf);
		err = true;
		exit(EXIT_FAILURE); 
	}
	else {
		unsigned long long one_uul = 1;
		int count = 0, d;
		char c = 1;
		c = getc(fileName);
			// the file starts with new line
			// second row: ? k n q current_number_of_matrix
			// second row is already read in the main function
			// the elements of the generator matrix are not devided for q<10
			// the deivider gor q>10 is ','
			// if any other symbol is entered the function returns error
			//printf("Reading data form file...\n");
				while ((c != '\r') && (c != '\n')) {
					c = getc(fileName);
				}
				if (q < 10) {
					for (int i = 1; i <= k; i++) {	//matrix indexing from [1][0]
						for (int j = 0; j < n; j++) {
							c = getc(fileName);
							if ((c < 48) || (c > 57)) return true;
							if (c == '0')
								d = 0;
							if (c == '1')
								d = 1;
							if (c == '2')
								d = 2;
							if (c == '3')
								d = 3;
							if (c == '4')
								d = 4;
							if (c == '5')
								d = 5;
							if (c == '6')
								d = 6;
							if (c == '7')
								d = 7;
							if (c == '8')
								d = 8;
							if (c == '9')
								d = 9;
							matrix.a[i][j] = d;

						}
						if (i <= k) {
							c = 1;
							int counter = 0;
							while ((c != '\r') && (c != '\n')) {
								c = getc(fileName);
								counter++;
								if (counter > 2 * n) {
									return true;
								}
							}

						}

					}
				}
				else {
					char numberString[2];
					int stringIt = 0;
					for (int i = 1; i <= k; i++) { //matrix indexing from [1][0]
						for (int j = 0; j < n; j++) {
							c = getc(fileName);
							if ((c < 48) || (c > 57)) return true;
							numberString[0] = c;
							c = getc(fileName);
							if (c == ',') {
								numberString[1] = numberString[0];
								numberString[0] = '0';

							}
							else if ((c < 48) || (c > 57)) {return true;}
							else {
								numberString[1] = c;
								c = getc(fileName);
								//printf("%c\n", c);
							}
							
							matrix.a[i][j] = charStriingToInt(numberString, 2);
							//cout << matrix[i][j] << "  ";
							//cout << numberString[0]<<numberString[1] << "  ";
						}
						//cout << endl;
						if (i < k) {
							c = 1;
							while ((c != '\r') && (c != '\n')) {
								c = getc(fileName);
							}

						}
					}
					c = getc(fileName);
				}



		
	}
	
	return err;
}

void printMatrix(bool form, char* file){
	FILE* out = fopen(file,"a");
	//ofstream out;
	//out.open(file, ios::app);
	if (out != NULL) {
		if (form) { fprintf(out,"! "); }
		else { fprintf(out, "? "); }
		fprintf(out, "%d %d %d %d\n",matrix.k, matrix.n,matrix.q,matrix.num);
		if (matrix.q > 9) {
			for (int i = 1; i <= matrix.k; i++) {
				for (int j = 0; j < matrix.n; j++) {
					fprintf(out, "%llu,", matrix.a[i][j]);
					//out << matrix.a[i][j] << ",";
				}
				fprintf(out, "\n");
				//out << endl;
			}
		}
		else {
			for (int i = 1; i <= matrix.k; i++) {
				for (int j = 0; j < matrix.n; j++) {
					fprintf(out, "%llu", matrix.a[i][j]);
					//out << matrix.a[i][j];
				}
				fprintf(out, "\n");
				//out << endl;
			}
		}

		fprintf(out, "\n");
		fclose(out);
		//out << endl;
		//out.close();
	}
	else {
		FILE* err = fopen("error.txt", "w");
		fprintf(err, "cannot open file!!\n");
		fclose(err);
		exit(EXIT_FAILURE);
	}
}


void printWeights(unsigned long long int* weights, int N, char* file) {
	//ofstream out;
	//out.open(file, ios::app);
	//out << "Weight distribution:\n";
	FILE* out = fopen(file, "a");
	if (out != NULL) {
		fprintf(out, "Weight distribution : \n");
		for (int i = 0; i <= N; i++) {
			//printf("%d words with weight %d\n", weightsGF2[i], i);
			if (weights[i] > 0) {
				//weights[i] = weights[i] * (Q - 1);
				//out << i << "^" << weights[i] << "   ";
				fprintf(out,"%d^%llu   ", i, weights[i]);
				//weights[i] = 0;
			}

		}
		fprintf(out, "\n\n");
		fclose(out);
		//out << endl << endl;
		//out.close();
	}
	else {
		FILE* err = fopen("error.txt", "w");
		fprintf(err, "cannot open file!!\n");
		fclose(err);
		exit(EXIT_FAILURE);
	}

}


void randomgenf(int n, int k, int q, int num)
{  // time_t t;

	  /* Intializes random number generator */
	time_t t;

	/* Intializes random number generator */
	srand((unsigned)time(&t));

	FILE* fran;
	//q=5;
	fran = fopen("EXAM", "a"); // append
	// Print n random numbers.
	int ki, ni;
	fprintf(fran, "\n");
	//for (numi = 1; numi <= num; numi++)
	//{
	fprintf(fran, "? %d %d %d %d \n", k, n, q, num);
	for (ki = 1; ki <= k; ki++) // starts indexing from [1]
	{
		for (ni = 0; ni < n; ni++) {
			if (ni+1 > k) {
				int qh;
				if ((ni+1) == (k + 1)) { qh = 1; }
				else { qh = rand() % q; }
				if (q > 10) {
					fprintf(fran, "%d,", qh);
					matrix.a[ki][ni] = qh;
				}
				else {
					fprintf(fran, "%d", qh);
					matrix.a[ki][ni] = qh;
				}
			}
			else {
				if (ni+1 == ki) {
					if (q > 10) {
						fprintf(fran, "1,");
						matrix.a[ki][ni] = 1;
					}
					else {
						fprintf(fran, "1");
						matrix.a[ki][ni] = 1;
					}
				}
				else {
					if (q > 10) {
						fprintf(fran, "0,");
						matrix.a[ki][ni] = 0;
					}
					else {
						fprintf(fran, "0");
						matrix.a[ki][ni] = 0;
					}
				}
			}
		}
		fprintf(fran, "\n");
	}
	//}
	int i = fclose(fran);
};


void randomgenf(int n, int k, int q, int **matrix, bool multiplicativeForm)
{  
	if (multiplicativeForm && q != 4 && q != 8 && q != 16 && q != 32 && q != 64 && q != 9 && q != 27 && q != 25 && q != 49) {
		
		printf("Multiplicative form can be used only for composite fields!\nNo generation!");
		return;
	}
	  /* Intializes random number generator */
	time_t t;

	/* Intializes random number generator */
	srand((unsigned)time(&t));

	FILE* fran;
	//q=5;
	fran = fopen("EXAM", "w");
	// Print n random numbers.
	int ki, ni;
	fprintf(fran, "\n");
	//for (numi = 1; numi <= num; numi++)
	//{
	
	if (multiplicativeForm) { fprintf(fran, "! %d %d %d %d \n", k, n, q, 1); }
	else { fprintf(fran, "? %d %d %d %d \n", k, n, q, 1); }
	for (ki = 0; ki < k; ki++) // from 0 to k
	{
		for (ni = 0; ni < n; ni++) { // from 0 to n
			if (ni > k) {
				int qh;
				if (multiplicativeForm) {
					if (ni == k) { qh = 0; } // ni == k
					else { qh = rand() % (q-1); }
				}
				else {
					if (ni == k) { qh = 1; } 
					else { qh = rand() % q; }
				}

				if (q > 10) {
					fprintf(fran, "%d,", qh);
					matrix[ki][ni] = qh;
				}
				else {
					fprintf(fran, "%d", qh);
					matrix[ki][ni] = qh;
				}
			}
			else {
				if (ni == ki) {
					if (multiplicativeForm) {
						if (q > 10) {
							fprintf(fran, "0,");
							matrix[ki][ni] = 0;
						}
						else {
							fprintf(fran, "0");
							matrix[ki][ni] = 0;
						}
					}
					else {
						if (q > 10) {
							fprintf(fran, "1,");
							matrix[ki][ni] = 1;
						}
						else {
							fprintf(fran, "1");
							matrix[ki][ni] = 1;
						}
					}

				}
				else {
					if (multiplicativeForm) {

						if (q > 10) {
							fprintf(fran, "%d,",q);
							matrix[ki][ni] = q;
						}
						else {
							fprintf(fran, "%d",q);
							matrix[ki][ni] = q;
						}
					}
					else {

						if (q > 10) {
							fprintf(fran, "0,");
							matrix[ki][ni] = 0;
						}
						else {
							fprintf(fran, "0");
							matrix[ki][ni] = 0;
						}
					}

				}
			}
		}
		fprintf(fran, "\n");
	}
	//}
	int i = fclose(fran);
};

