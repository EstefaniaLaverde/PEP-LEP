#include <iostream>
#include <fstream>
#include "lib256.h"


#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#include <cpuid.h>
#endif

using namespace std;

#if defined(__AVX2__)

static union {
    __m128i reg128_matrix_GF2[K_GF2][N_GF2 / 2];
    __m256i reg256_matrix_GF2[K_GF2][N_GF2 / 4];
    unsigned long long int matrix_GF2[K_GF2][N_GF2]; // for GF2 and GF4; bitwise representation of the elements of the field with characteristic 2
    __m128i reg128_matrix_CH2[K_GF2][N_CH2 / 16];
    __m256i reg256_matrix_CH2[K_GF2][N_CH2 / 32];
    unsigned char matrix_CH2[K_GF2][N_CH2]; // for GF8, GF16, GF32, GF64; bytewise representation of the elements ofthe field with characteristic 2
    unsigned long long int matrix_CH3[K_CH3][N_CH3]; // for fields with characteristic 3; bitwise representation of the elements
    __m128i reg128_matrix_CH3[K_CH3][N_CH3 / 2];
    __m256i reg256_matrix_CH3[K_CH3][N_CH3 / 4];
    unsigned char matrix_p[K_P][N_P]; // for ohter finite fields (GF5, GF7, GF11, ..., GF25, ..., GF49, ...); bytewise representation of the elements
    __m128i reg128_matrix_p[K_P][N_P / 16];
    __m256i reg256_matrix_p[K_P][N_P / 32];
};

// array that save the current linear combination
// used to calculate naext linear combination
static union {
    __m256i reg256_helper_GF2[K_GF2][N_GF2 / 4];
    __m128i reg128_helper_GF2[K_GF2][N_GF2 / 2];
    unsigned long long int helper_GF2[K_GF2][N_GF2];
    unsigned char helper_CH2[K_GF2][N_CH2];
    __m256i reg256_helper_CH2[K_GF2][N_CH2 / 32];
    __m128i reg128_helper_CH2[K_GF2][N_CH2 / 16];
    unsigned long long int helper_CH3[K_CH3][N_CH3];
    __m256i reg256_helper_CH3[K_CH3][N_CH3 / 4];
    __m128i reg128_helper_CH3[K_CH3][N_CH3 / 2];
    unsigned char helper_p[K_P][N_P];
    __m256i reg256_helper_p[K_P][N_P / 32];
    __m128i reg128_helper_p[K_P][N_P / 16];
};

static __m256i zero;
static __m256i Q_reg_Bytes;

//----------------------------- additional writing functions for 256-bit registers -------------------------//

void write_GF2_256(int res) {
    if (file != NULL) {
        int c = ((N - 1) / 64) + 1;
        unsigned long long int one = 1;
        for (int el = 0; el < c; el++) {
            if ((el) * 64 > N) {
                fprintf(file, "\n");
                return;
            }
            for (int shift = 0; shift < 64; shift++) {
                if (el * 64 + shift > (N - 1)) {
                    fprintf(file, "\n");
                    return;
                }
                if (helper_GF2[res][el] & (one << (63 - shift))) {
                    fprintf(file, "%d", 1);
                }
                else {
                    fprintf(file, "%d", 0);
                }

            }
        }
        fprintf(file, "\n");
    }
}

void write_GF2_coset_64(int res, int el) {
    if (file != NULL) {
        unsigned long long int one = 1;
        for (int shift = 0; shift < N; shift++) {
            if (helper_GF2[res][el] & (one << (63 - shift))) {
                fprintf(file, "%d", 1);
            }
            else {
                fprintf(file, "%d", 0);
            }

        }

        fprintf(file, "\n");
    }
}

void write_GF2_coset_256(int res, int el) {
    if (file != NULL) {
        unsigned long long int one = 1;
        for (int shift = 0; shift < 64; shift++) {
            if (0 * 64 + shift > (N - 1)) {
                fprintf(file, "\n");
                return;
            }
            if (helper_GF2[res][el] & (one << (63 - shift))) {
                fprintf(file, "%d", 1);
            }
            else {
                fprintf(file, "%d", 0);
            }
        }

        for (int shift = 0; shift < 64; shift++) {
            if ((0+1) * 64 + shift > (N - 1)) {
                fprintf(file, "\n");
                return;
            }
            if (helper_GF2[res][el + 1] & (one << (63 - shift))) {
                fprintf(file, "%d", 1);
            }
            else {
                fprintf(file, "%d", 0);
            }

        }

        fprintf(file, "\n");
    }
}


void write_ByteCH2_256(int res) {
    if (file != NULL) {
        for (int i = 0; i < N; i++) {
            int t = (int)helper_CH2[res][i];//helper_p
            if (form) {
                write_multpl(t, file);
            }
            else {
                if (Q > 9) { fprintf(file, "%d,", t); }
                else { fprintf(file, "%d", t); }
                //fprintf(file, "%d, ", t);
            }

        }
        fprintf(file, "\n");
    }
}

//write the codeword at position in file res for GF25 and GF49
void write_CF_256(int res) {
    if (file != NULL) {
        int t = 0;
        int ch = 5;
        if (Q == 49) ch = 7;
        int shift = (((N - 1) / 32) + 1);
        for (int i = 0; i < N; i++) {
            t = 0;
            t = (ch * helper_p[res][i]) + helper_p[res][i + 32 * shift];
            if (form) {
                write_multpl(t, file);
            }
            else {
                 fprintf(file, "%d,", t);
            }
        }
        fprintf(file, "\n");
    }
}

void write_Bytes_256(int res) {
    if (file != NULL) {
        for (int i = 0; i < N; i++) {
            int t = (int)helper_p[res][i];
            if (form) {
                write_multpl(t, file);
            }
            else {
                if (Q > 9) { fprintf(file, "%d,", t); }
                else { fprintf(file, "%d", t); }
                //fprintf(file, "%d, ", t);
            }

        }
        fprintf(file, "\n");
    }
}




void write_GF3_256(int res) {
    if (file != NULL) {
        int c = (((N - 1) / 64) + 1);
        int bit1 = 0;
        if (c < 3 || c % 4 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 4 * register_elements;
        }
        unsigned long long int one = 1;
        for (int i = 0; i < c; i++) {
            if ((i) * 64 > N) {
                fprintf(file, "\n");
                return;
            }
            for (int shift = 0; shift < 64; shift++) {
                if (i * 64 + shift > (N - 1)) {
                    fprintf(file, "\n");
                    return;
                }
                bool first, second;
                if (c == 2) {
                    first = helper_CH3[res][i * c] & (one << (63 - shift));
                    second = helper_CH3[res][i * c + 1] & (one << (63 - shift));
                }
                else {
                    first = helper_CH3[res][i] & (one << (63 - shift));
                    second = helper_CH3[res][i + bit1] & (one << (63 - shift));
                }

                if (first && second) { fprintf(file, "%d", 0); }
                else if (first) { fprintf(file, "%d", 1); }
                else if (second) { fprintf(file, "%d", 2); }
                else { printf("ERROR in writing in file for GF3 - element is 00!\n\n\n"); return; }
            }
        }
        fprintf(file, "\n");
    }
}

void write_GF9_256(int res) {
    if (file != NULL) {
        int c = (((N - 1) / 64) + 1);

        int bit1 = 0;
        if (c <3 || c % 4 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 4 * register_elements;
        }
        unsigned long long int one = 1;
        for (int i = 0; i < c; i++) {
            if ((i) * 64 > N) {
                fprintf(file, "\n");
                return;
            }
            for (int shift = 0; shift < 64; shift++) {
                int temp = 0;
                int result = 0;
                if (i * 64 + shift > (N - 1)) {
                    fprintf(file, "\n");
                    return;
                }
                unsigned long long int first = 0, second = 0;
                if (c == 2) {
                    first = helper_CH3[res][i * bit1] & (one << (63 - shift));
                    second = helper_CH3[res][i * bit1 + 4] & (one << (63 - shift));
                }
                else {
                    first = helper_CH3[res][i] & (one << (63 - shift));
                    second = helper_CH3[res][i + bit1] & (one << (63 - shift));
                }

                if (first != 0 && second != 0) { temp = 0; }
                else if (first != 0) { temp = 1; }
                else if (second != 0) { temp = 2; }
                else { printf( "EROR in writing in file for GF9 - element is 00!\n\n\n"); return; }
                result = result + temp;

                if (c == 2) {
                    first = helper_CH3[res][i * bit1 + 1] & (one << (63 - shift));
                    second = helper_CH3[res][i * bit1 + 5] & (one << (63 - shift));
                }
                else {
                    first = helper_CH3[res][i + 2 * bit1] & (one << (63 - shift));
                    second = helper_CH3[res][i + 3 * bit1] & (one << (63 - shift));
                }
                if (first != 0 && second != 0) { temp = 0; }
                else if (first != 0) { temp = 1; }
                else if (second != 0) { temp = 2; }
                else { printf("EROR in writing in file for GF9 - element is 00!\n\n\n"); return; }
                result = result + 3 * temp;

                if (form) {
                    write_multpl(result, file);
                }
                else {
                    fprintf(file, "%d", result);
                }
            }
        }
        fprintf(file, "\n");
    }
}

void write_GF27_256(int res) {
    if (file != NULL) {
        int c = (((N - 1) / 64) + 1);

        int bit1 = 0;
        if (c <3 || c % 4 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 4 * register_elements;
        }
        unsigned long long int one = 1;


                int temp = 0;
                int result = 0;
                bool first, second;

                if (c == 2) {
                    for (int shift = 0; shift < 64; shift++) {
                         temp = 0;
                         result = 0;
                        if (0 * 64 + shift > (N - 1)) { fprintf(file, "\n"); return; }
                        //^0
                        first = helper_CH3[res][0] & (one << (63 - shift));
                        second = helper_CH3[res][1] & (one << (63 - shift));
                        if (first && second) { temp = 0; }
                        else if (first) { temp = 1; }
                        else if (second) { temp = 2; }
                        else { printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                        result = result + temp;
                        //^1
                        first = helper_CH3[res][4] & (one << (63 - shift));
                        second = helper_CH3[res][5] & (one << (63 - shift));
                        if (first && second) { temp = 0; }
                        else if (first) { temp = 1; }
                        else if (second) { temp = 2; }
                        else {printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                        result = result + 3 * temp;
                        //^2
                        first = helper_CH3[res][8] & (one << (63 - shift));
                        second = helper_CH3[res][9] & (one << (63 - shift));
                        if (first && second) { temp = 0; }
                        else if (first) { temp = 1; }
                        else if (second) { temp = 2; }
                        else { printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                        result = result + 9 * temp;
                        if (form) {
                            write_multpl(result, file);
                        }
                        else {
                            fprintf(file, "%d,", result);
                        }
                    }

                    for (int shift = 0; shift < 64; shift++) {
                         temp = 0;
                         result = 0;
                        if (1 * 64 + shift > (N - 1)) { fprintf(file, "\n"); return; }
                        //^0
                        first = helper_CH3[res][2] & (one << (63 - shift));
                        second = helper_CH3[res][3] & (one << (63 - shift));
                        if (first && second) { temp = 0; }
                        else if (first) { temp = 1; }
                        else if (second) { temp = 2; }
                        else { printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                        result = result + temp;
                        //^1
                        first = helper_CH3[res][6] & (one << (63 - shift));
                        second = helper_CH3[res][7] & (one << (63 - shift));
                        if (first && second) { temp = 0; }
                        else if (first) { temp = 1; }
                        else if (second) { temp = 2; }
                        else { printf( "EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                        result = result + 3 * temp;
                        //^2
                        first = helper_CH3[res][10] & (one << (63 - shift));
                        second = helper_CH3[res][11] & (one << (63 - shift));
                        if (first && second) { temp = 0; }
                        else if (first) { temp = 1; }
                        else if (second) { temp = 2; }
                        else { printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                        result = result + 9 * temp;
                        if (form) {
                            write_multpl(result, file);
                        }
                        else {
                            fprintf(file, "%d,", result);
                        }
                    }



                    
                }
                else {

                    for (int i = 0; i < c; i++) {
                        if ((i) * 64 > N) {
                            fprintf(file, "\n");
                            return;
                        }
                        for (int shift = 0; shift < 64; shift++) {
                            temp = 0;
                            result = 0;
                            if (i * 64 + shift > (N - 1)) { fprintf(file, "\n"); return; }
                            //^0
                            first = helper_CH3[res][i] & (one << (63 - shift));
                            second = helper_CH3[res][i + 1 * bit1] & (one << (63 - shift));
                            if (first && second) { temp = 0; }
                            else if (first) { temp = 1; }
                            else if (second) { temp = 2; }
                            else {  printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                            result = result + temp;
                            //^1
                            first = helper_CH3[res][i + 2 * bit1] & (one << (63 - shift));
                            second = helper_CH3[res][i + 3 * bit1] & (one << (63 - shift));
                            if (first && second) { temp = 0; }
                            else if (first) { temp = 1; }
                            else if (second) { temp = 2; }
                            else { printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                            result = result + 3 * temp;
                            //^2
                            first = helper_CH3[res][i + 4 * bit1] & (one << (63 - shift));
                            second = helper_CH3[res][i + 5 * bit1] & (one << (63 - shift));
                            if (first && second) { temp = 0; }
                            else if (first) { temp = 1; }
                            else if (second) { temp = 2; }
                            else { printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
                            result = result + 9 * temp;
                            if (form) {
                                write_multpl(result, file);
                            }
                            else {
                                fprintf(file, "%d,", result);
                            }

                        }

                    }
                }
            
        
        fprintf(file, "\n");
    }
}

void write_CF2_256(int res) {
    if (file != NULL) {
        unsigned long long int one = 1;
        int c = ((N - 1) / 64) + 1;
        int bit1 = 0;
        if (c < 3 || c % 4 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 4 * register_elements;
        }
        for (int el = 0; el < c; el++) {
            if ((el) * 64 > N) { fprintf(file, "\n"); return; }
            for (int shift = 0; shift < 64; shift++) {
                int result = 0;
                if (el * 64 + shift > (N - 1)) { fprintf(file, "\n");  return; }
                for (int m = 0; m < M; m++) {
                    if (helper_GF2[res][el + m * bit1] & (one << (63 - shift))) {
                        result = result + (1 << m);
                    }
                }
                if (form) {
                    write_multpl(result, file);
                }
                else {
                    if (Q > 9) { fprintf(file, "%d,", result); }
                    else { fprintf(file, "%d", result); }
                    //fprintf(file, "%d, ", result);
                }
            }
        }
        fprintf(file, "\n");
    }
}

//---------------------------- additional writing functions for 256-bit register ------------------------//
//--------------------------------Byte Representation-------------------------------------//

//sets registers for byte representation of the elements
void setRegistersBytes_256(dmat_type& bits) {
    register_elements = (((N - 1) / 256) + 1);
    zero = _mm256_setzero_si256();
    Q_reg_Bytes = _mm256_set_epi8((char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q,
        (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q,
        (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q);

    if (Q % 2 == 0) {

        for (int i = 0; i < K_GF2; i++) {
            for (int j = 0; j < N_CH2; j++) {
                matrix_CH2[i][j] = 0;
                helper_CH2[i][j] = 0;
            }
        }

        for (int i = 1; i <= (M * K); i++) {
            for (int j = 0; j < N; j++) {
                matrix_CH2[i][j] = bits.a[i - 1][j];
            }
        }
    }
    else {
        for (int i = 0; i < K_P; i++) {
            for (int j = 0; j < N_P; j++) {
                matrix_p[i][j] = 0;
                helper_p[i][j] = 0;
            }
        }

        for (int i = 1; i <= (M * K); i++) {
            for (int j = 0; j < N; j++) {
                matrix_p[i][j] = bits.a[i - 1][j];
                //cout <<(int) matrix_p[i][j] << " ";

            }
            //cout << endl;
        }
    }

}

void setRegistersCF_256(dmat_type& bits) {

    register_elements = (((N - 1) / 256) + 1);
    zero = _mm256_setzero_si256();
    Q_reg_Bytes = _mm256_set_epi8((char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, 
        (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic,
        (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, 
        (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic,
        (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, 
        (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic);

    
    for (int row = 0; row < K_P; row++) {
        for (int col = 0; col < N_P; col++) {
            matrix_p[row][col] = 0;
            helper_p[row][col] = 0;
        }
    }
    int shift = (((N - 1) / 32) + 1); 
    for (int row = 1; row <= (M * K); row++) {
        for (int col = 0; col < N; col++) {
            matrix_p[row][col] = bits.a[row - 1][col];
            matrix_p[row][col + 32 * shift] = bits.a[row - 1][col + N];
        }
    }
}

static inline void add_256(int rec, int i, int res) {
    __m256i res_add, res_sub;
    for (int col = 0; col < (((N - 1) / 32) + 1); col++) {
        res_add = _mm256_add_epi8(reg256_helper_p[rec][col], reg256_matrix_p[i][col]);
        res_sub = _mm256_sub_epi8(res_add, Q_reg_Bytes);
        reg256_helper_p[res][col] = _mm256_blendv_epi8(res_sub, res_add, res_sub);
    }

}



static inline void add_CH2_256(int rec, int i, int res) {
    for (int col = 0; col < ((N - 1) / 32) + 1; col++) {
        reg256_helper_CH2[res][col] = _mm256_xor_si256(reg256_helper_CH2[rec][col], reg256_matrix_CH2[i][col]);
    }
}

static inline void add_CF_256(int rec, int i, int res) {
    __m256i res_add, res_sub;
    int shift = (((N - 1) / 32) + 1);
    for (int col = 0; col < shift; col++) {
        // ^0
        res_add = _mm256_add_epi8(reg256_helper_p[rec][col], reg256_matrix_p[i][col]);
        res_sub = _mm256_sub_epi8(res_add, Q_reg_Bytes);
        reg256_helper_p[res][col] = _mm256_blendv_epi8(res_sub, res_add, res_sub);

        // ^1
        res_add = _mm256_add_epi8(reg256_helper_p[rec][col + shift], reg256_matrix_p[i][col + shift]);
        res_sub = _mm256_sub_epi8(res_add, Q_reg_Bytes);
        reg256_helper_p[res][col + shift] = _mm256_blendv_epi8(res_sub, res_add, res_sub);

    }
}

static inline unsigned long long int weight_256(int res) {
    __m256i r0, r2, r3, zero256 = _mm256_setzero_si256();;
    __m128i r5;
    __m256i h = _mm256_set_epi8(8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
    unsigned long long int w = 0;
    for (int col = 0; col < (((N - 1) / 32) + 1); col++) {
        r0 = _mm256_cmpeq_epi8(reg256_helper_p[res][col], zero256);
        r2 = _mm256_and_si256(r0, h);
        r3 = _mm256_bsrli_epi128(r2, 8);
        r0 = _mm256_or_si256(r2, r3);
        r5 = _mm_or_si128(_mm256_castsi256_si128(r0), _mm256_extractf128_si256(r0, 1));
        unsigned long long int* t = (unsigned long long int*) & r5;
        w = w + ((32 - popcount(t[0])));
    }
    return w;
}


static inline unsigned long long int weight_CH2_256(int result) {
    __m256i r0, r2, r3,  r6;
    __m128i r5;
    __m256i h = _mm256_set_epi8(8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
    unsigned long long int w = 0;
    for (int col = 0; col < (((N - 1) / 32) + 1); col++) {
        r0 = _mm256_cmpeq_epi8(reg256_helper_CH2[result][col], zero);
        r2 = _mm256_and_si256(r0, h);
        r3 = _mm256_bsrli_epi128(r2, 8);
        r0 = _mm256_or_si256(r2, r3);
        r5 = _mm_or_si128(_mm256_castsi256_si128(r0), _mm256_extractf128_si256(r0, 1));
        unsigned long long int* t = (unsigned long long int*) & r5;
        w = w + ((32 - popcount(t[0])));
    }
    return w;
}


static inline unsigned long long int weight_CF_256(int res) {
   __m256i res_add0, res_add1;
    __m256i r0, r2, r3, r6, zero256 = _mm256_setzero_si256();;
    __m128i r5;
    __m256i h = _mm256_set_epi8(8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
    unsigned long long int w = 0;
     int shift = (((N - 1) / 32) + 1);
      for (int col = 0; col < shift; col++) {
        res_add0 = _mm256_cmpeq_epi8(reg256_helper_p[res][col], zero);
        res_add1 = _mm256_cmpeq_epi8(reg256_helper_p[res][col + shift], zero);
        r0 = _mm256_and_si256(res_add0, res_add1);
        r2 = _mm256_and_si256(r0, h);
        r3 = _mm256_bsrli_epi128(r2, 8);
        r0 = _mm256_or_si256(r2, r3);
        r5 = _mm_or_si128(_mm256_castsi256_si128(r0), _mm256_extractf128_si256(r0, 1));
        unsigned long long int* t = (unsigned long long int*) & r5;
        w = w + ((32 - popcount(t[0])));
    }
    return w;
    
   /* unsigned long long int w = 0;
    int shift = (((N - 1) / 32) + 1);
    for (int col = 0; col < shift; col++) {
        res_add0 = _mm256_cmpgt_epi8(reg256_helper_p[res][col], zero);
        res_add1 = _mm256_cmpgt_epi8(reg256_helper_p[res][col + shift], zero);

        __m256i weight_reg = _mm256_or_si256(res_add0, res_add1);

        unsigned long long int* t = (unsigned long long int*) & weight_reg;
        w = w + ((popcount(t[0]) + popcount(t[1]) + popcount(t[2]) + popcount(t[3])) >> 3);
    }*/
    return w;
}


void linear_combinations_Bytes_256(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_256(rec - 1, i, rec);
                unsigned long long int w = weight_256(rec);
                weights[w]++;
            }
            else {
                add_256(rec, i, rec);
                unsigned long long int w = weight_256(rec);
                weights[w]++;
            }

            if (rec < K) {
                linear_combinations_Bytes_256(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CH2_256(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                unsigned long long int weight = 0;
                add_CH2_256(rec - 1, i, rec);
                weight = weight_CH2_256(rec);
                weights[weight]++;
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                unsigned long long int weight = 0;
                add_CH2_256(rec, t * K + i, rec);
                weight = weight_CH2_256(rec);
                weights[weight]++;
            }
            if (rec < K) {
                linear_combinations_CH2_256(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_Bytes_256_euqal_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_256(rec - 1, i, rec);
            }
            else {
                add_256(rec, i, rec);
            }

            unsigned long long int w = weight_256(rec);
            if ((w == w_searched)) {
                write_Bytes_256(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_Bytes_256_euqal_count(rec + 1, i + 1);
            }
        }
    }

}


void linear_combinations_CH2_256_equal_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CH2_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CH2_256(rec, t * K + i, rec);
            }

            unsigned long long int weight = 0;
            weight = weight_CH2_256(rec);
            if ((weight == w_searched)) {
                write_ByteCH2_256(rec);
            }
            weights[weight]++;

            if (rec < K) {
                linear_combinations_CH2_256_equal_count(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CF_49_256_equal_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence49[q1] - 1;
                add_CF_256(rec, t * K + i, rec);
            }

            unsigned long long int weight = 0;
            weight = weight_CF_256(rec);
            if ((weight == w_searched)) {
                write_CF_256(rec);
            }
            weights[weight]++;

            if (rec < K) {
                linear_combinations_CF_49_256_equal_count(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CF_25_256_equal_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF_256(rec - 1, i, rec);
            }
            else {
                short t = TransitionSequence25[q1] - 1;
                add_CF_256(rec, t * K + i, rec);
            }
            unsigned long long int weight = 0;
            weight = weight_CF_256(rec);
            if ((weight == w_searched)) {
                write_CF_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_CF_25_256_equal_count(rec + 1, i + 1);
            }
        }
    }
}



void linear_combinations_Bytes_256_equal_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_256(rec - 1, i, rec);
            }
            else {
                add_256(rec, i, rec);
            }

            unsigned long long int w = weight_256(rec);
            if ((w == w_searched)) {
                write_Bytes_256(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_Bytes_256_equal_count(rec + 1, i + 1);
            }
        }
    }

}



void linear_combinations_Bytes_256_less_than_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_256(rec - 1, i, rec);
            }
            else {
                add_256(rec, i, rec);
            }

            unsigned long long int w = weight_256(rec);
            if ((w < w_searched)) {
                write_Bytes_256(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_Bytes_256_less_than_count(rec + 1, i + 1);
            }
        }
    }

}

void linear_combinations_CH2_256_less_than_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CH2_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CH2_256(rec, t * K + i, rec);
            }

            unsigned long long int weight = 0;
            weight = weight_CH2_256(rec);
            if ((weight < w_searched)) {
                write_ByteCH2_256(rec);
            }
            weights[weight]++;

            if (rec < K) {
                linear_combinations_CH2_256_less_than_count(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CF_49_256_less_than_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence49[q1] - 1;
                add_CF_256(rec, t * K + i, rec);
            }

            unsigned long long int weight = 0;
            weight = weight_CF_256(rec);
            if ((weight < w_searched)) {
                write_CF_256(rec);
            }
            weights[weight]++;

            if (rec < K) {
                linear_combinations_CF_49_256_less_than_count(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CF_25_256_less_than_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF_256(rec - 1, i, rec);
            }
            else {
                short t = TransitionSequence25[q1] - 1;
                add_CF_256(rec, t * K + i, rec);
            }
            unsigned long long int weight = 0;
            weight = weight_CF_256(rec);
            if ((weight < w_searched)) {
                write_CF_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_CF_25_256_less_than_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_CF_25_256(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF_256(rec - 1, i, rec);
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
            }
            else {
                short t = TransitionSequence25[q1] - 1;
                add_CF_256(rec, t * K + i, rec);
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
            }
            if (rec < K) {
                linear_combinations_CF_25_256(rec + 1, i + 1);
            }
        }
    }

}

void linear_combinations_CF_49_256(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF_256(rec - 1, i, rec);
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
            }
            else {
                short t = TransitionSequence49[q1] - 1;
                add_CF_256(rec, t * K + i, rec);
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
            }
            if (rec < K) {
                linear_combinations_CF_49_256(rec + 1, i + 1);
            }
        }
    }

}


void linear_combinations_Bytes_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_256(rec - 1, i, rec);
                }
                else {
                    add_256(rec, i, rec);
                }
                unsigned long long int w = weight_256(rec);
                weights[w]++;
                if (w < w_searched) {
                    less_than_flag = false;
                }
                if (rec < K) {
                    linear_combinations_Bytes_256_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_CH2_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2_256(rec, t * K + i, rec);
                }
                unsigned long long int w = 0;
                w = weight_CH2_256(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CH2_256_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_CF_25_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF_256(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    add_CF_256(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
                if (w < w_searched) {
                    less_than_flag = false;
                }

                if (rec < K) {
                    linear_combinations_CF_25_256_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_CF_49_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF_256(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    add_CF_256(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
                if (w < w_searched) {
                    less_than_flag = false;
                }

                if (rec < K) {
                    linear_combinations_CF_49_256_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_Bytes_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_256(rec - 1, i, rec);
                }
                else {
                    add_256(rec, i, rec);
                }
                unsigned long long int w = weight_256(rec);
                weights[w]++;
                if (w == w_searched) {
                    equal_flag = false;
                }
                if (rec < K) {
                    linear_combinations_Bytes_256_equal(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_CH2_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2_256(rec, t * K + i, rec);
                }
                unsigned long long int w = 0;
                w = weight_CH2_256(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CH2_256_equal(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_CF_25_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF_256(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    add_CF_256(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
                if (w == w_searched) {
                    equal_flag = false;
                }

                if (rec < K) {
                    linear_combinations_CF_25_256_equal(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_CF_49_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF_256(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    add_CF_256(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF_256(rec);
                weights[w]++;
                if (w == w_searched) {
                    equal_flag = false;
                }

                if (rec < K) {
                    linear_combinations_CF_49_256_equal(rec + 1, i + 1);
                }
            }
        }
    }
}


//------------------------------------  GF2  ----------------------------------------------//

//setting data in registers for n<=64
void set_64_256(dynamic_mat_short& bits) {
    reg256_matrix_GF2[0][0] = _mm256_setzero_si256();
    reg256_helper_GF2[0][0] = _mm256_setzero_si256();


    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        reg256_matrix_GF2[row][0] = _mm256_setzero_si256();
        reg256_helper_GF2[row][0] = _mm256_setzero_si256();


        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][0];
        matrix_GF2[row][2] = bits.a[row - 1][0];
        matrix_GF2[row][3] = bits.a[row - 1][0];

    }

    //using coset for calculation of two codewords at the same time
    helper_GF2[0][1] = bits.a[K - 1][0];
    helper_GF2[0][2] = bits.a[K - 2][0];
    helper_GF2[0][3] = bits.a[K - 1][0] ^ bits.a[K - 2][0];

}

void set_128_256(dynamic_mat_short& bits) {
    reg256_matrix_GF2[0][0] = _mm256_setzero_si256();
    reg256_helper_GF2[0][0] = _mm256_setzero_si256();


    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        reg256_matrix_GF2[row][0] = _mm256_setzero_si256();
        reg256_helper_GF2[row][0] = _mm256_setzero_si256();


        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][1];
        matrix_GF2[row][2] = bits.a[row - 1][0];
        matrix_GF2[row][3] = bits.a[row - 1][1];

    }

    //using coset for calculation of two codewords at the same time
    helper_GF2[0][2] = bits.a[K - 1][0];
    helper_GF2[0][3] = bits.a[K - 1][1];

}

// setting data in registers for n>64 && n<=128


// setting data into registers for n>256
void set_256(dynamic_mat_short& bits) {
    register_elements = (((N - 1) / 256) + 1);
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }
    for (int col = 0; col < register_elements; col++) {
        reg256_matrix_GF2[0][col] = _mm256_setzero_si256();
        reg256_helper_GF2[0][col] = _mm256_setzero_si256();
    }
    for (int row = 1; row <= K; row++) {
        for (int col = 0; col < (N_GF2 / 4); col++) {
            reg256_matrix_GF2[row][col] = _mm256_setzero_si256();
            reg256_helper_GF2[row][col] = _mm256_setzero_si256();
        }
        for (int el = 0; el < N_GF2; el++) {
            if ((el) * 64 > N) {
                break;
            }
            matrix_GF2[row][el] = bits.a[row - 1][el];
        }
    }
}

// addition for GF2 n < 256
static inline void add_GF2_256_256(int rec, int i, int res) {
    reg256_helper_GF2[res][0] = _mm256_xor_si256(reg256_helper_GF2[rec][0], reg256_matrix_GF2[i][0]);
}



//------------------------------- functions for calculations for n<=64 --------------------------------//
void linear_combinations_64_256_less_than(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j < K-1; j++) {
            add_GF2_256_256(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;

            w = popcount(helper_GF2[rec][1]);
            if (w < w_searched) {
                less_than_flag = false;
            }
            w = popcount(helper_GF2[rec][2]);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;

            w = popcount(helper_GF2[rec][3]);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;
            if (rec < K - 2) {
                linear_combinations_64_256_less_than(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_64_256_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j < K-1; j++) {
            add_GF2_256_256(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            w = popcount(helper_GF2[rec][1]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            w = popcount(helper_GF2[rec][2]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            w = popcount(helper_GF2[rec][3]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            if (rec < K - 2) {
                linear_combinations_64_256_equal(rec + 1, j + 1);
            }
        }
    }
}


void linear_combinations_64_256_less_than_count(int rec, int h) {
    for (int j = h; j < K-1; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]);
        if (w < w_searched) {
            write_GF2_coset_64(rec, 0);
        }
        weights[w]++;
        w = 0;
        w = popcount(helper_GF2[rec][1]);
        if (w < w_searched) {
            write_GF2_coset_64(rec, 1);
        }
        weights[w]++;
        w = popcount(helper_GF2[rec][2]);
        if (w < w_searched) {
            write_GF2_coset_64(rec, 2);
        }
        weights[w]++;
        w = 0;
        w = popcount(helper_GF2[rec][3]);
        if (w < w_searched) {
            write_GF2_coset_64(rec, 3);
        }
        weights[w]++;
        if (rec < K - 2) {
            linear_combinations_64_256_less_than_count(rec + 1, j + 1);
        }
    }
}

void linear_combinations_64_256_equal_count(int rec, int h) {
    for (int j = h; j < K-1; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]);
        if (w == w_searched) {
            write_GF2_coset_64(rec, 0);
        }
        weights[w]++;

        w = popcount(helper_GF2[rec][1]);
        if (w == w_searched) {
            write_GF2_coset_64(rec, 1);
        }
        weights[w]++;

        w = popcount(helper_GF2[rec][2]);
        if (w == w_searched) {
            write_GF2_coset_64(rec, 2);
        }
        weights[w]++;

        w = popcount(helper_GF2[rec][3]);
        if (w == w_searched) {
            write_GF2_coset_64(rec, 3);
        }
        weights[w]++;

        if (rec < K - 2) {
            linear_combinations_64_256_equal_count(rec + 1, j + 1);
        }
    }
}



void linear_combinations_64_256(int rec, int h) {
    for (int j = h; j < (K-1); j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]);
        weights[w]++;

        w = popcount(helper_GF2[rec][1]);
        weights[w]++;

        w = popcount(helper_GF2[rec][2]);
        weights[w]++;

        w = popcount(helper_GF2[rec][3]);
        weights[w]++;
        if (rec < K - 2) {
            linear_combinations_64_256(rec + 1, j + 1);
        }
    }
}

//----------------------------functions for 64<n<=128--------------------------------------------------//

void linear_combinations_128_256_less_than(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j < K; j++) {
            add_GF2_256_256(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;

            w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
            if (w < w_searched) {
                less_than_flag = false;
            }

            weights[w]++;
            if (rec < K - 1) {
                linear_combinations_128_256_less_than(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_128_256_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j < K; j++) {
            add_GF2_256_256(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;
            if (rec < K - 1) {
                linear_combinations_128_256_equal(rec + 1, j + 1);
            }
        }
    }
}


void linear_combinations_128_256_less_than_count(int rec, int h) {
    for (int j = h; j < K; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
        if (w < w_searched) {
            //new function for coset for n>64
            write_GF2_coset_256(rec, 0);
        }
        weights[w]++;
        w = 0;
        w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        if (w < w_searched) {
            //new function for coset for n>64
            write_GF2_coset_256(rec, 2);
        }
        weights[w]++;
        if (rec < K - 1) {
            linear_combinations_128_256_less_than_count(rec + 1, j + 1);
        }
    }
}

void linear_combinations_128_256_equal_count(int rec, int h) {
    for (int j = h; j < K; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
        if (w == w_searched) {
            //new function for coset for n>64
            write_GF2_coset_256(rec, 0);
        }
        weights[w]++;

        w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        if (w == w_searched) {
            //new function for coset for n>64
            write_GF2_coset_256(rec, 2);
        }
        weights[w]++;


        if (rec < K - 1) {
            linear_combinations_128_256_equal_count(rec + 1, j + 1);
        }
    }
}



void linear_combinations_128_256(int rec, int h) {
    for (int j = h; j < K; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
        weights[w]++;

        w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        weights[w]++;

        if (rec < K - 1) {
            linear_combinations_128_256(rec + 1, j + 1);
        }
    }
}

//=====================================================================================================//

// function for calculation the weight for GF2 n>64 && n<=128
static inline unsigned long long int weight_GF2_256_256(int res) {
    unsigned long long int w = 0;
    w = popcount(helper_GF2[res][0]) + popcount(helper_GF2[res][1]) + popcount(helper_GF2[res][2]) + popcount(helper_GF2[res][3]);
    return w;
}

// ---------------------------- functions for GF2 128 < n <=256 ------------------------------------//
void linear_combinations_256_256_less_than(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_256_256(rec - 1, j, rec);
            unsigned int w = weight_GF2_256_256(rec);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_256_256_less_than(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_256_256_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_256_256(rec - 1, j, rec);
            unsigned int w = weight_GF2_256_256(rec);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_256_256_equal(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_256_256_less_than_count(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned int w = weight_GF2_256_256(rec);
        if (w < w_searched) {
            write_GF2_256(rec);
        }
        weights[w]++;

        if (rec < K) {
            linear_combinations_256_256_less_than_count(rec + 1, j + 1);
        }
    }
}

void linear_combinations_256_256_equal_count(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned int w = weight_GF2_256_256(rec);
        if (w == w_searched) {
            write_GF2_256(rec);
        }
        weights[w]++;

        if (rec < K) {
            linear_combinations_256_256_equal_count(rec + 1, j + 1);
        }
    }
}

void linear_combinations_256_256(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_256_256(rec - 1, j, rec);
        unsigned int w = weight_GF2_256_256(rec);
        weights[w]++;

        if (rec < K) {
            linear_combinations_256_256(rec + 1, j + 1);
        }
    }

}


static inline void add_GF2_256(int rec, int i, int res) {
    unsigned long long int w = 0;
    unsigned long long int temp[4];
    //unsigned long long int* temp;
    for (int el = 0; el < register_elements; el++) {
        reg256_helper_GF2[res][el] = _mm256_xor_si256(reg256_helper_GF2[rec][el], reg256_matrix_GF2[i][el]);
       // w = w + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el], 0)) + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el], 1))
       //     + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el], 2)) + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el], 3));
        //_mm256_storeu_si256((__m256i*) & temp[0], reg256_helper_GF2[res][el]);
       // temp = (unsigned long long int*) & reg256_helper_GF2[res][el];
         //w = w + popcount(temp[0]) + popcount(temp[1]) + popcount(temp[2]) + popcount(temp[3]);
         w = w + popcount(helper_GF2[res][4 * el]) + popcount(helper_GF2[res][4 * el + 1])
            + popcount(helper_GF2[res][4 * el + 2]) + popcount(helper_GF2[res][4 * el + 3]);
    }
    weights[w]++;
}


static inline unsigned long long int weight_GF2_256(int res) {
    unsigned long long int w = 0;
    unsigned long long int temp[64];
    for (int el = 0; el < register_elements; el++) {
        //_mm256_storeu_si256((__m256i*) & temp[0], reg256_helper_GF2[res][el]);
        //temp = (unsigned long long*) & reg256_helper_GF2[res][el];
        // w = w + popcount(temp[0]) + popcount(temp[1])
    // + popcount(temp[2]) + popcount(temp[3]);
        w = w + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el],0)) + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el], 1))
            + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el], 2)) + popcount(_mm256_extract_epi64(reg256_helper_GF2[res][el], 3));
       // w = w + popcount(helper_GF2[res][4 * el]) + popcount(helper_GF2[res][4 * el + 1])
       //     + popcount(helper_GF2[res][4 * el + 2]) + popcount(helper_GF2[res][4 * el + 3]);
    }
    return w;
}


void linear_combinations_256_less_than(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_256(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_256(rec);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_256_less_than(rec + 1, j + 1);
            }
        }
    }

}

void linear_combinations_256_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_256(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_256(rec);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_256_equal(rec + 1, j + 1);
            }
        }
    }

}


void linear_combinations_256_less_than_count(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_256(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_256(rec);
        if (w < w_searched) {
            write_GF2_256(rec);
        }
        //weights[w]++;
        if (rec < K) {
            linear_combinations_256_less_than_count(rec + 1, j + 1);
        }
    }
}

void linear_combinations_256_equal_count(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_256(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_256(rec);
        if (w == w_searched) {
            write_GF2_256(rec);
        }
        //weights[w]++;
        if (rec < K) {
            linear_combinations_256_equal_count(rec + 1, j + 1);
        }
    }
}


void linear_combinations_256(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_256(rec - 1, j, rec);
        //unsigned long long int w = weight_GF2_256(rec);
        //weights[w]++;
        if (rec < K) {
            linear_combinations_256(rec + 1, j + 1);
        }
    }
}



//------------------------------------  End GF2  ----------------------------------------------//



//---------------------------------Characteristic 2 ---------------------------------------//

void setMatrixGF2_CF_256(dynamic_mat_short& bits) {
    for (int i = 0; i <= N_FIX * 8; i++) {
        weights[i] = 0;
    }
    register_elements = ((N - 1) / 256) + 1;
    int c = ((N - 1) / 64) + 1;
    int bit1 = 0;
    if (c < 3 || c % 4 == 0) {
        bit1 = c;
    }
    else {
        bit1 = 4 * register_elements;
    }
    for (int col = 0; col < (N_GF2 / 4); col++) {
        reg256_matrix_GF2[0][col] = _mm256_setzero_si256();
        reg256_helper_GF2[0][col] = _mm256_setzero_si256();
    }

    for (int row = 1; row <= (M * (K + 1)); row++) {
        for (int col = 0; col < (N_GF2 / 4); col++) {
            reg256_matrix_GF2[row][col] = _mm256_setzero_si256();
            reg256_helper_GF2[row][col] = _mm256_setzero_si256();
        }
        for (int el = 0; el < c; el++) {
            if ((el) * 64 > N) {
                break;
            }

            for (int m = 0; m < M; m++) {
                matrix_GF2[row][el + m * bit1] = bits.a[row - 1][el + m * c];

            }
        }
    }

}

static inline void add_CF2_256(int rec, int i, int res) {
    for (int reg = 0; reg < register_elements; reg++) {
        for (int m = 0; m < M; m++) {
            reg256_helper_GF2[res][m * register_elements + reg] =
                _mm256_xor_si256(reg256_helper_GF2[rec][m * register_elements + reg], reg256_matrix_GF2[i][m * register_elements + reg]);
        }
    }
}



unsigned long long int weight_GF4_64(int res, int pos) {
    unsigned long long int* temp = (unsigned long long int*) & reg128_helper_GF2[res][pos];
    unsigned long long  w = temp[0] | temp[1];
    int weight = popcount(w);
    return weight;
}

unsigned long long int weight_GF4_128(int res) {

    unsigned long long  int *temp;
    __m128i temp_reg = _mm_setzero_si128();
    temp_reg = _mm_or_si128(reg128_helper_GF2[res][0], reg128_helper_GF2[res][1]);
    temp = (unsigned long long int*) & temp_reg;
    int weight = popcount(temp[0]) + popcount(temp[1]);
    return weight;
}

static inline unsigned long long int weight_CF2_256(int res) {
    __m256i temp_reg = _mm256_setzero_si256();
    int w = 0;
    for (int reg = 0; reg < register_elements; reg++) {
        temp_reg = _mm256_setzero_si256();
        for (int m = 0; m < M; m++) {
            temp_reg = _mm256_or_si256(temp_reg, reg256_helper_GF2[res][m * register_elements + reg]);
        }
        unsigned long long* temp = (unsigned long long*) & temp_reg;
        w = w + popcount(temp[0]) + popcount(temp[1]) + popcount(temp[2]) + popcount(temp[3]);
    }
    return w;
}

void linear_combinations_CF2_256(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF2_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CF2_256(rec, t * K + i, rec);
            }
            unsigned long long int w = weight_CF2_256(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_CF2_256(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_64_256(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                add_GF2_256_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF2_256_256(rec, t * K + i, rec);
            }
            
            unsigned long long int weight = weight_GF4_64(rec,0);
            weights[weight]++;
            
            //weight = weight_GF4_64(rec, 2);
            //weights[weight]++;

            if (rec < K) {
                linear_combinations_GF4_64_256(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_GF4_128_256(int rec, int h) {

    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_GF2_256_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF2_256_256(rec, t * K + i, rec);
            }
            int weight = weight_GF4_128(rec);
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF4_128_256(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_64_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF2_256_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF2_256_256(rec, t * K + i, rec);
                }
                unsigned long long int weight = weight_GF4_64(rec, 0);
                weights[weight]++;
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                if (rec < K) {
                    linear_combinations_GF4_64_256_less_than(rec + 1, i + 1);
                }
            }
        }
    }

}
void linear_combinations_GF4_128_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF2_256_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF2_256_256(rec, t * K + i, rec);
                }
                int weight = weight_GF4_128(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_128_256_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_CF2_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF2_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CF2_256(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF2_256(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CF2_256_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_GF4_64_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF2_256_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF2_256_256(rec, t * K + i, rec);
                }
                unsigned long long int weight = weight_GF4_64(rec,0);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;

                if (rec < K) {
                    linear_combinations_GF4_64_256_equal(rec + 1, i + 1);
                }
            }
        }
    }

}


void linear_combinations_GF4_128_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF2_256_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF2_256_256(rec, t * K + i, rec);
                }
                int weight = weight_GF4_128(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_128_256_equal(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_CF2_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF2_256(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CF2_256(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF2_256(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CF2_256_equal(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_GF4_64_256_equal_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_GF2_256_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF2_256_256(rec, t * K + i, rec);
            }
            unsigned long long int weight = weight_GF4_64(rec,0);
            if (weight == w_searched) {
                write_CF2_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF4_64_256_equal_count(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_GF4_128_256_equal_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_GF2_256_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF2_256_256(rec, t * K + i, rec);
            }
            int weight = weight_GF4_128(rec);
            if (weight == w_searched) {
                write_CF2_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF4_128_256_equal_count(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CF2_256_equal_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF2_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CF2_256(rec, t * K + i, rec);
            }
            unsigned long long int w = weight_CF2_256(rec);
            if (w == w_searched) {
                write_CF2_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_CF2_256_equal_count(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_GF4_64_256_less_than_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_GF2_256_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF2_256_256(rec, t * K + i, rec);
            }
            unsigned long long int weight = weight_GF4_64(rec,0);
            if (weight < w_searched) {
                write_CF2_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF4_64_256_less_than_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_128_256_less_than_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_GF2_256_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF2_256_256(rec, t * K + i, rec);
            }
            int weight = weight_GF4_128(rec);
            if (weight < w_searched) {
                write_CF2_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF4_128_256_less_than_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_CF2_256_less_than_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CF2_256(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CF2_256(rec, t * K + i, rec);
            }
            unsigned long long int w = weight_CF2_256(rec);
            if (w < w_searched) {
                write_CF2_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_CF2_256_less_than_count(rec + 1, i + 1);
            }
        }
    }
}

//------------------------------------------End Characteristic 2 -------------------------------------//




// -----------------------------  Characteristic 3 --------------------------------------------//

void setMatrixGF3_256(dynamic_mat_short &bits) {
    int c = (((N - 1) / 64) + 1);
    int bit1 = 0;

    // e.g. if we need 3 64-bit computer words for the given n, we will need 2 128-bit registers
    // writing scheme:
    // |    first bit of the representation    |    second bit of the representation   |
    // | 64 bits | 64 bits | 64 bits | ------- | 64 bits | 64 bits | 64 bits | ------- |
    // |      128 bits     |      128 bits     |      128 bits     |      128 bits     |

    if (c < 3 || c % 4 == 0) { 
        bit1 = c;
    }
    else {
        bit1 = 4 * register_elements;
    }

    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3/4; el++) {
        reg256_matrix_CH3[0][el] = _mm256_setzero_si256();
        reg256_helper_CH3[0][el] = _mm256_set1_epi64x(zero);//all 1 vector
    }

    for (int row = 1; row <= K; row++) {
        for (int i = 0; i < N_CH3 / 4; i++) {
            reg256_matrix_CH3[row][i] = _mm256_setzero_si256();
            reg256_helper_CH3[row][i] = _mm256_setzero_si256();
        }
        if (N > 64 && N <= 128) {
            matrix_CH3[row][0] = bits.a[row - 1][0];
            matrix_CH3[row][1] = bits.a[row - 1][c];
            matrix_CH3[row][2] = bits.a[row - 1][1];
            matrix_CH3[row][3] = bits.a[row - 1][1 + c];
        }
        else {
            for (int el = 0; el < c; el++) {
                matrix_CH3[row][el] = bits.a[row - 1][el];
                matrix_CH3[row][el + bit1] = bits.a[row - 1][el + c];
                if (N <= 64) {
                    matrix_CH3[row][el + 2] = bits.a[row - 1][el];
                    matrix_CH3[row][el + bit1 + 2] = bits.a[row - 1][el + c];
                }
            }
        }

    }
    if (N <= 64) {
        helper_CH3[0][2] = bits.a[K - 1][0];
        helper_CH3[0][3] = bits.a[K - 1][1];
    }
}


void setMatrixGF9_256(dynamic_mat_short& bits) {
    int c = (((N - 1) / 64) + 1);

    int bit1 = 0;
    if (c <3|| c % 4 == 0) {
        bit1 = c;
    }
    else {
        bit1 = 4 * register_elements;
    }


    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3 / 4; el++) {
        reg256_matrix_CH3[0][el] = _mm256_setzero_si256();
        reg256_helper_CH3[0][el] = _mm256_set1_epi64x(zero);//all 1 vector
    }

    for (int row = 1; row <= 2 * K; row++) {
        for (int i = 0; i < N_CH3 / 4; i++) {
            reg256_matrix_CH3[row][i] = _mm256_setzero_si256();
            reg256_helper_CH3[row][i] = _mm256_setzero_si256();
        }

        if (c == 2) {
           // for (int el = 0; el < c; el++) {
                // ^0
             /* matrix_CH3[row][0] = bits.a[row - 1][0];
                matrix_CH3[row][1] = bits.a[row - 1][2 * c];

                matrix_CH3[row][2] = bits.a[row - 1][1];
                matrix_CH3[row][3] = bits.a[row - 1][1 + 2 * c];

                // ^1
                matrix_CH3[row][4] = bits.a[row - 1][ c];
                matrix_CH3[row][5] = bits.a[row - 1][3 * c];

                matrix_CH3[row][6] = bits.a[row - 1][1 + c];
                matrix_CH3[row][7] = bits.a[row - 1][1 + 3 * c];
                */
            // I bit
            matrix_CH3[row][0] = bits.a[row - 1][0]; //^0
            matrix_CH3[row][1] = bits.a[row - 1][c]; //^1

            matrix_CH3[row][2] = bits.a[row - 1][1]; //^0
            matrix_CH3[row][3] = bits.a[row - 1][1 + c]; //^1

            // II bit
            matrix_CH3[row][4] = bits.a[row - 1][2 * c]; //^0
            matrix_CH3[row][5] = bits.a[row - 1][3 * c]; //^1

            matrix_CH3[row][6] = bits.a[row - 1][1 + 2 * c]; //^0
            matrix_CH3[row][7] = bits.a[row - 1][1 + 3 * c]; //^1

            //}
        }
        else {
            
            for (int el = 0; el < c; el++) {
                // ^0
                matrix_CH3[row][el] = bits.a[row - 1][el];
                matrix_CH3[row][el + 1 * bit1] = bits.a[row - 1][el + 2 * c];
                // ^1
                matrix_CH3[row][el + 2 * bit1] = bits.a[row - 1][el + c];
                matrix_CH3[row][el + 3 * bit1] = bits.a[row - 1][el + 3 * c];

               

            }
        }


    }
}

void setMatrixGF27_256(dynamic_mat_short& bits) {

    int c = (((N - 1) / 64) + 1);
    register_elements = ((N - 1) / 256) + 1;
    int bit1 = 0;
    if (c < 3 || c % 4 == 0) {
        bit1 = c;
    }
    else {
        bit1 = 4 * register_elements;
    }

    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3 / 4; el++) {
        reg256_matrix_CH3[0][el] = _mm256_setzero_si256();
        reg256_helper_CH3[0][el] = _mm256_set1_epi64x(zero);//all 1 vector
    }

    for (int row = 1; row <= 3 * K; row++) {
        for (int i = 0; i < N_CH3 / 4; i++) {
            reg256_matrix_CH3[row][i] = _mm256_setzero_si256();
            reg256_helper_CH3[row][i] = _mm256_setzero_si256();
        }

        if (c == 2) {
            // ^0
            matrix_CH3[row][0] = bits.a[row - 1][0]; // bit 0
            matrix_CH3[row][0 + 1 * bit1] = bits.a[row - 1][1];  // bit 0
            // ^1
            matrix_CH3[row][0 + 2 * bit1] = bits.a[row - 1][0 + c];  // bit 0
            matrix_CH3[row][0 + 3 * bit1] = bits.a[row - 1][1 + c];   // bit 0
            // ^2
            matrix_CH3[row][0 + 4 * bit1] = bits.a[row - 1][0 + 2 * c];  // bit 0
            matrix_CH3[row][0 + 5 * bit1] = bits.a[row - 1][1 + 2 * c];  // bit 0

            // ^0
            matrix_CH3[row][1] = bits.a[row - 1][0 + 4 * c];  // bit1
            matrix_CH3[row][1 + 1 * bit1] = bits.a[row - 1][1 + 4 * c]; // bit1
            // ^1
            matrix_CH3[row][1 + 2 * bit1] = bits.a[row - 1][0 + 5 * c];  // bit1
            matrix_CH3[row][1 + 3 * bit1] = bits.a[row - 1][1 + 5 * c];  // bit 1
            // ^2
            matrix_CH3[row][1 + 4 * bit1] = bits.a[row - 1][0 + 6 * c];  // bit1
            matrix_CH3[row][1 + 5 * bit1] = bits.a[row - 1][1 + 6 * c];  // bit1

        }
        else {
            for (int el = 0; el < c; el++) {
                // ^0
                matrix_CH3[row][el] = bits.a[row - 1][el]; // bit 0
                matrix_CH3[row][el + 1 * bit1] = bits.a[row - 1][el + 4 * c]; // bit1
                // ^1
                matrix_CH3[row][el + 2 * bit1] = bits.a[row - 1][el + c];  // bit 0
                matrix_CH3[row][el + 3 * bit1] = bits.a[row - 1][el + 5 * c];  // bit 1
                // ^2
                matrix_CH3[row][el + 4 * bit1] = bits.a[row - 1][el + 2 * c];  // bit 0
                matrix_CH3[row][el + 5 * bit1] = bits.a[row - 1][el + 6 * c];  // bit1

            }
        }

    }
}

static inline void add_GF3_64_256(int j, int rec, int res) { //?
    __m256i xor_1 = _mm256_setzero_si256();
    __m256i xor_2 = _mm256_setzero_si256();
    __m256i xor_rev = _mm256_setzero_si256();

    __m256i xor_rev2 = _mm256_setzero_si256();

    xor_1 = _mm256_xor_si256(reg256_matrix_CH3[j][0], reg256_helper_CH3[rec][0]);

    xor_rev2 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][0], 78);
    xor_2 = _mm256_xor_si256(xor_1, xor_rev2);


    xor_rev = _mm256_shuffle_epi32(xor_1, 78);
    // xor_rev = _mm_castpd_si128(_mm_permute_pd(
    //     _mm_castsi128_pd(xor_1), (int)1
    // ));
    reg256_helper_CH3[res][0] = _mm256_or_si256(xor_2, xor_rev);
}

static inline unsigned long long int weight_GF3_64_256(int res, int pos) {
    unsigned long long int w = 0;
    unsigned long long  w_and = helper_CH3[res][pos] ^ helper_CH3[res][pos+1];
    w = popcount(w_and);
    return w;
}
static inline unsigned long long int weight_GF3_128_256(int res) {//?
    unsigned long long int w = 0;
    static union {
        __m128i w_xor = _mm_setzero_si128();
        unsigned long long  w_xor64[2];
    };
    w_xor = _mm_xor_si128(reg128_helper_CH3[res][0], reg128_helper_CH3[res][1]);
    w = popcount(w_xor64[0]) + popcount(w_xor64[1]);
    return w;
}

void linear_comb_recGF3_64_256(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_256(j, rec - 2 + q1, rec);
            int w = weight_GF3_64_256(rec,0);
            weights[w]++;
           // w = weight_GF3_64_256(rec, 2);
            //weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_64_256(rec + 1, j + 1);
            }
        }

    }
}


void linear_comb_recGF3_128_256(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_256(j, rec - 2 + q1, rec);
            int w = weight_GF3_64_256(rec,0) + weight_GF3_64_256(rec,2);
            weights[w]++;

            if (rec < K ) {
                linear_comb_recGF3_128_256(rec + 1, j + 1);
            }
        }

    }
}
static inline void add_GF3_AVX(int j, int rec, int res) {//?
    __m256i xor_1[2];
    __m256i xor_2[2];


    for (int el = 0; el < register_elements; el++) {
        xor_1[0] = _mm256_xor_si256(reg256_matrix_CH3[j][el], reg256_helper_CH3[rec][el]);
        xor_1[1] = _mm256_xor_si256(reg256_matrix_CH3[j][el + register_elements], reg256_helper_CH3[rec][el + register_elements]);

        xor_2[0] = _mm256_xor_si256(xor_1[0], reg256_matrix_CH3[j][el + register_elements]);
        xor_2[1] = _mm256_xor_si256(xor_1[1], reg256_matrix_CH3[j][el]);

        reg256_helper_CH3[res][el] = _mm256_or_si256(xor_2[0], xor_1[1]);
        reg256_helper_CH3[res][el + register_elements] = _mm256_or_si256(xor_2[1], xor_1[0]);
    }

}

static inline unsigned long long int weight_GF3_AVX(int res) {//?
    unsigned long long int w = 0;
    static union {
        __m256i w_xor = _mm256_setzero_si256();
        unsigned long long  w_xor64[4];
    };
    for (int el = 0; el < register_elements; el++) {

        w_xor = _mm256_xor_si256(reg256_helper_CH3[res][el], reg256_helper_CH3[res][el + register_elements]);
        w = w + popcount(w_xor64[0]) + popcount(w_xor64[1])+ popcount(w_xor64[2]) + popcount(w_xor64[3]);
    }
    return w;
}

void linear_comb_recGF3_AVX(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_AVX(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_AVX(rec);
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_AVX(rec + 1, j + 1);
            }
        }

    }
}


static inline void addGF9_64_256(int j, int rec, int res) {
    __m256i xor_1 = _mm256_setzero_si256();
    __m256i xor_2 = _mm256_setzero_si256();
    __m256i xor_rev = _mm256_setzero_si256();
    __m256i xor_rev2 = _mm256_setzero_si256();

    xor_1 = _mm256_xor_si256(reg256_matrix_CH3[j][0], reg256_helper_CH3[rec][0]);
    xor_rev2 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][0], 78);
    xor_2 = _mm256_xor_si256(xor_1, xor_rev2);
    xor_rev = _mm256_shuffle_epi32(xor_1, 78);
    reg256_helper_CH3[res][0] = _mm256_or_si256(xor_2, xor_rev);

}


static inline void addGF9_128_256(int j, int rec, int res) {
    __m256i xor_1[2];// = _mm256_setzero_si256();
    __m256i xor_2[2];// = _mm256_setzero_si256();
   // __m256i xor_rev = _mm256_setzero_si256();
   // __m256i xor_rev2 = _mm256_setzero_si256();

    xor_1[0] = _mm256_xor_si256(reg256_matrix_CH3[j][0], reg256_helper_CH3[rec][0]);
    xor_1[1] = _mm256_xor_si256(reg256_matrix_CH3[j][1], reg256_helper_CH3[rec][1]);

    xor_2[0] = _mm256_xor_si256(reg256_matrix_CH3[j][1], xor_1[0]);
    xor_2[1] = _mm256_xor_si256(reg256_matrix_CH3[j][0], xor_1[1]);

    //xor_rev2 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][0], 78);
    //xor_2 = _mm256_xor_si256(xor_1, xor_rev2);
    //xor_rev = _mm256_shuffle_epi32(xor_1, 78);


    reg256_helper_CH3[res][0] = _mm256_or_si256(xor_2[0], xor_1[1]);
    reg256_helper_CH3[res][1] = _mm256_or_si256(xor_2[1], xor_1[0]);

}

static inline unsigned long long int weight_GF9_128_256(int res) {
    unsigned long long int weight;
    static union {
        __m256i elements_xor = _mm256_setzero_si256();
        unsigned long long int xor64[4];
    };
    elements_xor = _mm256_xor_si256(reg256_helper_CH3[res][0], reg256_helper_CH3[res][1]);
    weight = popcount(xor64[0] | xor64[1]) + popcount(xor64[2] | xor64[3]);

    return weight;
}

static inline unsigned long long int weight_GF9_64_256(int res) {
    unsigned long long  element1 = 0, element2 = 0;
    element1 = helper_CH3[res][0] ^ helper_CH3[res][1];
    element2 = helper_CH3[res][2] ^ helper_CH3[res][3];
    unsigned long long int weight = 0;
    weight = popcount(element1 | element2);

    return weight;
}

void linear_combinations_GF9_64_256(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_256(t * K + j, rec, rec);
            }
            unsigned long long weight = weight_GF9_64_256(rec);
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_64_256(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_256(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_256(t * K + j, rec, rec);
            }
            unsigned long long weight = weight_GF9_128_256(rec);
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_128_256(rec + 1, j + 1);
            }
        }
    }
}

static inline void addGF9_AVX(int j, int rec, int res) {
    __m256i xor_1[4];// = _mm_setzero_si128();
    __m256i xor_2[4];// = _mm_setzero_si128();

    xor_1[0] = _mm256_setzero_si256();
    xor_1[1] = _mm256_setzero_si256();
    xor_2[0] = _mm256_setzero_si256();
    xor_2[1] = _mm256_setzero_si256();

    xor_1[2] = _mm256_setzero_si256();
    xor_1[3] = _mm256_setzero_si256();
    xor_2[2] = _mm256_setzero_si256();
    xor_2[3] = _mm256_setzero_si256();

    for (int i = 0; i <register_elements; i++) {
        xor_1[0] = _mm256_xor_si256(reg256_matrix_CH3[j][i ], reg256_helper_CH3[rec][i ]);
        xor_1[1] = _mm256_xor_si256(reg256_matrix_CH3[j][i  + 1*register_elements], reg256_helper_CH3[rec][i + 1 * register_elements]);
        xor_1[2] = _mm256_xor_si256(reg256_matrix_CH3[j][i  + 2 * register_elements], reg256_helper_CH3[rec][i + 2 * register_elements]);
        xor_1[3] = _mm256_xor_si256(reg256_matrix_CH3[j][i  + 3 * register_elements], reg256_helper_CH3[rec][i + 3 * register_elements]);

        xor_2[0] = _mm256_xor_si256(xor_1[0], reg256_matrix_CH3[j][i + 1 * register_elements]);
        xor_2[1] = _mm256_xor_si256(xor_1[1], reg256_matrix_CH3[j][i]);
        xor_2[2] = _mm256_xor_si256(xor_1[2], reg256_matrix_CH3[j][i + 3 * register_elements]);
        xor_2[3] = _mm256_xor_si256(xor_1[3], reg256_matrix_CH3[j][i + 2 * register_elements]);

        reg256_helper_CH3[res][i] = _mm256_or_si256(xor_2[0], xor_1[1]);
        reg256_helper_CH3[res][i + 1 * register_elements] = _mm256_or_si256(xor_2[1], xor_1[0]);
        reg256_helper_CH3[res][i + 2 * register_elements] = _mm256_or_si256(xor_2[2], xor_1[3]);
        reg256_helper_CH3[res][i + 3 * register_elements] = _mm256_or_si256(xor_2[3], xor_1[2]);
    }
}

static inline unsigned long long int weight_GF9_AVX(int res) {
    //__m128i element1 = _mm_setzero_si128(), element2 = _mm_setzero_si128();
    static union {
        __m256i res256 = _mm256_setzero_si256();
        unsigned long long int res64[4];
    };
    unsigned long long int count = 0;
    __m256i temp1 = _mm256_setzero_si256();
    __m256i temp2 = _mm256_setzero_si256();
    for (int i = 0; i <register_elements; i++) {
        temp1 = _mm256_xor_si256(reg256_helper_CH3[res][i], reg256_helper_CH3[res][i + 1 * register_elements]);
        temp2 = _mm256_xor_si256(reg256_helper_CH3[res][i+2 * register_elements], reg256_helper_CH3[res][i + 3 * register_elements]);
        res256 = _mm256_or_si256(temp1, temp2);
        count = count + popcount(res64[0]) + popcount(res64[1]) + popcount(res64[2]) + popcount(res64[3]);
    }
    return count;
}


void linear_combinations_GF9_AVX(int rec, int h) {
    int qf = 8;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_AVX(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_AVX(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_AVX(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_AVX(rec + 1, j + 1);
            }
        }
    }

}

static inline void addGF27_64_256(int j, int rec, int res) {
    __m256i xor1 = _mm256_setzero_si256();
    __m256i xor2 = _mm256_setzero_si256();
    __m256i rev1 = _mm256_setzero_si256();
    __m256i rev2 = _mm256_setzero_si256();

    //^0 and ^1
    xor1 = _mm256_xor_si256(reg256_matrix_CH3[j][0], reg256_helper_CH3[rec][0]);
    rev1 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][0], 78);
    xor2 = _mm256_xor_si256(xor1, rev1);
    rev2 = _mm256_shuffle_epi32(xor1, 78);
    reg256_helper_CH3[res][0] = _mm256_or_si256(xor2, rev2);

    //^3
    xor1 = _mm256_xor_si256(reg256_matrix_CH3[j][1], reg256_helper_CH3[rec][1]);
    rev1 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][1], 78);
    xor2 = _mm256_xor_si256(xor1, rev1);
    rev2 = _mm256_shuffle_epi32(xor1, 78);
    reg256_helper_CH3[res][1] = _mm256_or_si256(xor2, rev2);

}
static inline unsigned long long int  weight_GF27_64_256(int res) {
    unsigned long long  element1 = 0, element2 = 0, element3 = 0;
    element1 = helper_CH3[res][0] ^ helper_CH3[res][1];
    element2 = helper_CH3[res][2] ^ helper_CH3[res][3];
    element3 = helper_CH3[res][4] ^ helper_CH3[res][5];

    unsigned long long int count = 0;
    count = popcount(element1 | element2 | element3);
    return count;
}


static inline void addGF27_128_256(int j, int rec, int res) {
    __m256i xor1 = _mm256_setzero_si256();
    __m256i xor2 = _mm256_setzero_si256();
    __m256i rev1 = _mm256_setzero_si256();
    __m256i rev2 = _mm256_setzero_si256();

    //^0 
    xor1 = _mm256_xor_si256(reg256_matrix_CH3[j][0], reg256_helper_CH3[rec][0]);
    rev1 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][0], 78);
    xor2 = _mm256_xor_si256(xor1, rev1);
    rev2 = _mm256_shuffle_epi32(xor1, 78);
    reg256_helper_CH3[res][0] = _mm256_or_si256(xor2, rev2);

    //^1
    xor1 = _mm256_xor_si256(reg256_matrix_CH3[j][1], reg256_helper_CH3[rec][1]);
    rev1 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][1], 78);
    xor2 = _mm256_xor_si256(xor1, rev1);
    rev2 = _mm256_shuffle_epi32(xor1, 78);
    reg256_helper_CH3[res][1] = _mm256_or_si256(xor2, rev2);

    //^2
    xor1 = _mm256_xor_si256(reg256_matrix_CH3[j][2], reg256_helper_CH3[rec][2]);
    rev1 = _mm256_shuffle_epi32(reg256_matrix_CH3[j][2], 78);
    xor2 = _mm256_xor_si256(xor1, rev1);
    rev2 = _mm256_shuffle_epi32(xor1, 78);
    reg256_helper_CH3[res][2] = _mm256_or_si256(xor2, rev2);

}
static inline unsigned long long int  weight_GF27_128_256(int res) {
    unsigned long long  element1 = 0, element2 = 0, element3 = 0;
    unsigned long long int count = 0;
    element1 = helper_CH3[res][0] ^ helper_CH3[res][1];
    element2 = helper_CH3[res][4] ^ helper_CH3[res][ 5];
    element3 = helper_CH3[res][8] ^ helper_CH3[res][ 9];
    count = count + popcount(element1 | element2 | element3);

    element1 = helper_CH3[res][2] ^ helper_CH3[res][3];
    element2 = helper_CH3[res][6] ^ helper_CH3[res][7];
    element3 = helper_CH3[res][10] ^ helper_CH3[res][11];
    count = count + popcount(element1 | element2 | element3);
    

    return count;
}

static inline void addGF27_AVX(int j, int rec, int res) {
    __m256i xor_1[2];// = _mm256_setzero_si256();
    __m256i xor_2[2];// = _mm256_setzero_si256();
    xor_1[0] = _mm256_setzero_si256();
    xor_1[1] = _mm256_setzero_si256();
    xor_2[0] = _mm256_setzero_si256();
    xor_2[1] = _mm256_setzero_si256();

    for (int i = 0; i < register_elements; i++) {
        xor_1[0] = _mm256_xor_si256(reg256_matrix_CH3[j][i], reg256_helper_CH3[rec][i]);
        xor_1[1] = _mm256_xor_si256(reg256_matrix_CH3[j][i + register_elements], reg256_helper_CH3[rec][i + register_elements]);
        xor_2[0] = _mm256_xor_si256(xor_1[0], reg256_matrix_CH3[j][i + register_elements]);
        xor_2[1] = _mm256_xor_si256(xor_1[1], reg256_matrix_CH3[j][i]);

        reg256_helper_CH3[res][i] = _mm256_or_si256(xor_2[0], xor_1[1]);
        reg256_helper_CH3[res][i + register_elements] = _mm256_or_si256(xor_2[1], xor_1[0]);

        xor_1[0] = _mm256_xor_si256(reg256_matrix_CH3[j][i + 2 * register_elements], reg256_helper_CH3[rec][i + 2 * register_elements]);
        xor_1[1] = _mm256_xor_si256(reg256_matrix_CH3[j][i + 3 * register_elements], reg256_helper_CH3[rec][i + 3 * register_elements]);

        xor_2[0] = _mm256_xor_si256(xor_1[0], reg256_matrix_CH3[j][i + 3 * register_elements]);
        xor_2[1] = _mm256_xor_si256(xor_1[1], reg256_matrix_CH3[j][i + 2 * register_elements]);

        reg256_helper_CH3[res][i + 2 * register_elements] = _mm256_or_si256(xor_2[0], xor_1[1]);
        reg256_helper_CH3[res][i + 3 * register_elements] = _mm256_or_si256(xor_2[1], xor_1[0]);

        xor_1[0] = _mm256_xor_si256(reg256_matrix_CH3[j][i + 4 * register_elements], reg256_helper_CH3[rec][i + 4 * register_elements]);
        xor_1[1] = _mm256_xor_si256(reg256_matrix_CH3[j][i + 5 * register_elements], reg256_helper_CH3[rec][i + 5 * register_elements]);

        xor_2[0] = _mm256_xor_si256(xor_1[0], reg256_matrix_CH3[j][i + 5 * register_elements]);
        xor_2[1] = _mm256_xor_si256(xor_1[1], reg256_matrix_CH3[j][i + 4 * register_elements]);
        reg256_helper_CH3[res][i + 4 * register_elements] = _mm256_or_si256(xor_2[0], xor_1[1]);
        reg256_helper_CH3[res][i + 5 * register_elements] = _mm256_or_si256(xor_2[1], xor_1[0]);
    }
}
static inline unsigned long long int  weight_GF27_AVX(int res) {
    static union {
        __m256i res256 = _mm256_setzero_si256();
        unsigned long long int res64[4];
    };
    unsigned long long int count = 0;
    __m256i temp1 = _mm256_setzero_si256();
    __m256i temp2 = _mm256_setzero_si256(); 
    __m256i temp3 = _mm256_setzero_si256();
    for (int i = 0; i < register_elements; i++) {
        temp1 = _mm256_xor_si256(reg256_helper_CH3[res][i], reg256_helper_CH3[res][i + 1 * register_elements]);
        temp2 = _mm256_xor_si256(reg256_helper_CH3[res][i + 2 * register_elements], reg256_helper_CH3[res][i + 3 * register_elements]);
        temp3 = _mm256_xor_si256(reg256_helper_CH3[res][i + 4 * register_elements], reg256_helper_CH3[res][i + 5 * register_elements]);
        res256 = _mm256_or_si256(temp1, temp2);
        res256 = _mm256_or_si256(res256, temp3);
        count = count + popcount(res64[0]) + popcount(res64[1]) + popcount(res64[2]) + popcount(res64[3]);
    }
    return count;
}


void linear_combinations_GF27_64_256(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_64_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_64_256(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_64_256(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_64_256(rec + 1, j + 1);
            }
        }
    }
}
void linear_combinations_GF27_128_256(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_128_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_128_256(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_128_256(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_128_256(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_AVX(int rec, int h) {

    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_AVX(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_AVX(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_AVX(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_AVX(rec + 1, j + 1);
            }
        }
    }
}
void linear_comb_recGF3_64_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_256(j, rec - 2 + q1, rec);
                int w = weight_GF3_64_256(rec, 0);
                if (w < w_searched) {
                    less_than_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_64_256_less_than(rec + 1, j + 1);
                }
            }

        }
    }

}

void linear_comb_recGF3_128_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_256(j, rec - 2 + q1, rec);
                int w = weight_GF3_64_256(rec, 0) + weight_GF3_64_256(rec, 2);
                if (w < w_searched) {
                    less_than_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_128_256_less_than(rec + 1, j + 1);
                }
            }

        }
    }

}

void linear_comb_recGF3_AVX_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_AVX(j, rec - 2 + q1, rec);
                int w = weight_GF3_AVX(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_AVX_less_than(rec + 1, j + 1);
                }
            }

        }
    }

}

void linear_combinations_GF9_64_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_256(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_256(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_64_256(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_64_256_less_than(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_128_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_128_256(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_128_256(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_128_256(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_128_256_less_than(rec + 1, j + 1);
                }
            }
        }
    }
}
void linear_combinations_GF9_AVX_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_AVX(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_AVX(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_AVX(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_AVX_less_than(rec + 1, j + 1);
                }
            }
        }
    }
}
void linear_combinations_GF27_64_256_less_than(int rec, int h) {
        if (less_than_flag) {
            int qf;
            if (h == 1) { qf = 1; }
            else { qf = 26; }
            for (int j = h; j <= K; j++) {
                for (int q1 = 1; q1 <= qf; q1++) {
                    if (q1 == 1) {
                        addGF27_64_256(j, rec - 1, rec);
                    }
                    else {
                        int t = TransitionSequence27[q1] - 1;
                        addGF27_64_256(t * K + j, rec, rec);
                    }
                    unsigned long long int w = weight_GF27_64_256(rec);
                    if (w < w_searched) {
                        less_than_flag = false;
                    }
                    weights[w]++;
                    if (rec < K) {
                        linear_combinations_GF27_64_256_less_than(rec + 1, j + 1);
                    }
                }
            }
        }
    
}
void linear_combinations_GF27_128_256_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_128_256(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_128_256(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_128_256(rec);
                weights[w]++;
                if (w < w_searched) {
                    less_than_flag = false;
                }
                if (rec < K) {
                    linear_combinations_GF27_128_256_less_than(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF27_AVX_less_than(int rec, int h) {
        if (less_than_flag) {
            int qf;
            if (h == 1) { qf = 1; }
            else { qf = 26; }
            for (int j = h; j <= K; j++) {
                for (int q1 = 1; q1 <= qf; q1++) {
                    if (q1 == 1) {
                        addGF27_AVX(j, rec - 1, rec);
                    }
                    else {
                        int t = TransitionSequence27[q1] - 1;
                        addGF27_AVX(t * K + j, rec, rec);
                    }
                    unsigned long long int w = weight_GF27_AVX(rec);
                    
                    if (w < w_searched) {
                        less_than_flag = false;
                    }
                    weights[w]++;
                    if (rec < K) {
                        linear_combinations_GF27_AVX_less_than(rec + 1, j + 1);
                    }
                }
            }
        }
}

void linear_comb_recGF3_64_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_256(j, rec - 2 + q1, rec);
                int w = weight_GF3_64_256(rec,0);
                if (w == w_searched) {
                    equal_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_64_256_equal(rec + 1, j + 1);
                }
            }

        }
    }

}
void linear_comb_recGF3_128_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_256(j, rec - 2 + q1, rec);
                int w = weight_GF3_64_256(rec, 0) + weight_GF3_64_256(rec, 2);
                if (w == w_searched) {
                    equal_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_128_256_equal(rec + 1, j + 1);
                }
            }

        }
    }
}
void linear_comb_recGF3_AVX_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_AVX(j, rec - 2 + q1, rec);
                int w = weight_GF3_AVX(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_AVX_equal(rec + 1, j + 1);
                }
            }

        }
    }

}
void linear_combinations_GF9_64_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_256(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_256(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_64_256(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_64_256_equal(rec + 1, j + 1);
                }
            }
        }
    }

}
void linear_combinations_GF9_128_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_128_256(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_128_256(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_128_256(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_128_256_equal(rec + 1, j + 1);
                }
            }
        }
    }

}
void linear_combinations_GF9_AVX_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_AVX(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_AVX(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_AVX(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_AVX_equal(rec + 1, j + 1);
                }
            }
        }
    }
}
void linear_combinations_GF27_64_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_64_256(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_64_256(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_64_256(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_64_256_equal(rec + 1, j + 1);
                }
            }
        }
    }
}
void linear_combinations_GF27_128_256_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_128_256(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_128_256(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_128_256(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_128_256_equal(rec + 1, j + 1);
                }
            }
        }
    }
}
void linear_combinations_GF27_AVX_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_AVX(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_AVX(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_AVX(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_AVX_equal(rec + 1, j + 1);
                }
            }
        }
    }
}


void linear_comb_recGF3_64_256_equal_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_256(j, rec - 2 + q1, rec);
            int w = weight_GF3_64_256(rec,0);
            if ((w == w_searched)) {
                write_GF3_256(rec);
            }

            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_64_256_equal_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_128_256_equal_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_256(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_64_256(rec, 0) + weight_GF3_64_256(rec, 2);
            weights[w]++;
            if ((w == w_searched)) {
                //new function
                write_GF3_256(rec);
            }
            if (rec < K) {
                linear_comb_recGF3_128_256_equal_count(rec + 1, j + 1);
            }
        }

    }

}

void linear_comb_recGF3_AVX_equal_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_AVX(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_AVX(rec);
            if (w == w_searched) {
                write_GF3_256(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_AVX_equal_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_combinations_GF9_64_256_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_256(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_64_256(rec);
            weights[w]++;
           
            if (w == w_searched) {
                write_GF9_256(rec);
            }
            if (rec < K) {
                linear_combinations_GF9_64_256_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_256_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_256(t * K + j, rec, rec);
            }
            unsigned long long weight = weight_GF9_128_256(rec);
            if (weight == w_searched) {
                //new function
                 write_GF9_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_128_256_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_AVX_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_AVX(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_AVX(t * K + j, rec, rec);
            }
            unsigned long long weight = weight_GF9_AVX(rec);
            if (weight == w_searched) {
                  write_GF9_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_AVX_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_64_256_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_64_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_64_256(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_64_256(rec);
            if (w == w_searched) {
                write_GF27_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_64_256_equal_count(rec + 1, j + 1);
            }
        }
    }
}
void linear_combinations_GF27_128_256_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_128_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_128_256(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_128_256(rec);
            if (w == w_searched) {
                //new function
                write_GF27_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_128_256_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_AVX_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_AVX(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_AVX(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_AVX(rec);
            if (w == w_searched) {
                write_GF27_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_AVX_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_comb_recGF3_64_256_less_than_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_256(j, rec - 2 + q1, rec);
            int w = weight_GF3_64_256(rec,0);
            if ((w < w_searched)) {
                write_GF3_256(rec);
            }

            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_64_256_less_than_count(rec + 1, j + 1);
            }
        }

    }
}
void linear_comb_recGF3_128_256_less_than_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_256(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_64_256(rec, 0) + weight_GF3_64_256(rec, 2);
            weights[w]++;
            if ((w < w_searched)) {
                //new function
                write_GF3_256(rec);
            }
            if (rec < K) {
                linear_comb_recGF3_128_256_less_than_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_AVX_less_than_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_AVX(j, rec - 2 + q1, rec);
            int w = weight_GF3_AVX(rec);
            if ((w < w_searched)) {
                 write_GF3_256(rec);
            }

            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_AVX_less_than_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_combinations_GF9_64_256_less_than_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_256(t * K + j, rec, rec);
            }
            unsigned long long weight = weight_GF9_64_256(rec);
            if (weight < w_searched) {
                write_GF9_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_64_256_less_than_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_256_less_than_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_256(t * K + j, rec, rec);
            }
            unsigned long long weight = weight_GF9_128_256(rec);
            if (weight < w_searched) {
                //new function 
                 write_GF9_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_128_256_less_than_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_AVX_less_than_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_AVX(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_AVX(t * K + j, rec, rec);
            }
            unsigned long long weight = weight_GF9_AVX(rec);
            if (weight < w_searched) {
                 write_GF9_256(rec);
            }
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_AVX_less_than_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_64_256_less_than_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_64_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_64_256(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_64_256(rec);
            if (w < w_searched) {
                write_GF27_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_64_256_less_than_count(rec + 1, j + 1);
            }
        }
    }
}
void linear_combinations_GF27_128_256_less_than_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_128_256(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_128_256(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_128_256(rec);
            if (w < w_searched) {
                //new function 
                write_GF27_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_128_256_less_than_count(rec + 1, j + 1);
            }
        }
    }
}
void linear_combinations_GF27_AVX_less_than_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF27_AVX(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF27_AVX(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF27_AVX(rec);
            if (w < w_searched) {
                write_GF27_256(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_AVX_less_than_count(rec + 1, j + 1);
            }
        }
    }
}
// ----------------------------- END Characteristic 3 --------------------------------------------//



void calculateWeightBytes_256(dmat_type& bits, int n, int k, int m, int q) {
    
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    less_than_flag = true;
    register_elements = ((n - 1) / 256) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes_256(bits);
            linear_combinations_CH2_256(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersCF_256(bits);
            linear_combinations_CF_49_256(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersCF_256(bits);
            linear_combinations_CF_25_256(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes_256(bits);//!!!!!!!!!!!
        linear_combinations_Bytes_256(1, 1);//!!!!!!!!!!!
    }


    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
}



void calculateWeightGF2_256(dynamic_mat_short& bits, int n, int k) {

    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
   // less_than_flag = true;

    register_elements = ((n - 1) / 256) + 1;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (n <= 64) {
        set_64_256(bits);
        unsigned int w = popcount(helper_GF2[0][1]);
        weights[w]++;

        w = popcount(helper_GF2[0][2]);
        weights[w]++;

        w = popcount(helper_GF2[0][3]);
        weights[w]++;
        linear_combinations_64_256(1, 1);
    }
    else if (n <= 128) {
        set_128_256(bits);
        unsigned int w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        weights[w]++;
        linear_combinations_128_256(1, 1);
    }
    else  {
        set_256(bits);
        linear_combinations_256(1, 1);
    }
    

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }

}


void calculateWeightCH2_256(dynamic_mat_short& bits, int n, int k, int m) {

    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    less_than_flag = true;
    register_elements = ((n - 1) / 256) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    setMatrixGF2_CF_256(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_256(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_256(1, 1);
        }
        else {
            linear_combinations_CF2_256(1, 1);
        }
        break;
    case 8:
        linear_combinations_CF2_256(1, 1);
        break;
    case 16:
        linear_combinations_CF2_256(1, 1);
        break;
    case 32:
        linear_combinations_CF2_256(1, 1);
        break;
    case 64:
        linear_combinations_CF2_256(1, 1);
        break;
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
}

void calculateWeightCH3_256(dynamic_mat_short& bits, int n, int k, int m) {
    popcnt_detect();

    K = k;
    N = n;
    M = m;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    register_elements = (((n - 1) / 256) + 1);

    if (m == 1) {
        Q = 3;
        setMatrixGF3_256(bits);
        if (n <= 64) {
            linear_comb_recGF3_64_256(1, 1);
        }
        else if (n <= 128) {
            linear_comb_recGF3_128_256(1, 1);
        }
        else {
            linear_comb_recGF3_AVX(1, 1);
        }


    }
    else if (m == 2) {
        Q = 9;
        setMatrixGF9_256(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_256(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF9_128_256(1, 1);
        }
        else {
            linear_combinations_GF9_AVX(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
        setMatrixGF27_256(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_256(1, 1);

        }
        else if (n <= 128) {
            linear_combinations_GF27_128_256(1, 1);
        }
        else {
            linear_combinations_GF27_AVX(1, 1);
        }

    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }

}


//-------------------------------------------Chaeck if there is a word we weight less than given w--------------------------------------------------------------//


bool calculateWeightBytes_256_less_than(dmat_type& bits, int n, int k, int m, int q, int d) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    less_than_flag = true;
    register_elements = ((n - 1) / 256) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes_256(bits);
            linear_combinations_CH2_256_less_than(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersCF_256(bits);
            linear_combinations_CF_49_256_less_than(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersCF_256(bits);
            linear_combinations_CF_25_256_less_than(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes_256(bits);
        linear_combinations_Bytes_256_less_than(1, 1);
    }


    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}



bool calculateWeightGF2_256_less_than(dynamic_mat_short& bits, int n, int k, int d) {
    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    less_than_flag = true;

    register_elements = ((n - 1) / 256) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }



    if (n <= 64) {
        set_64_256(bits);
        unsigned int w = popcount(helper_GF2[0][1]);
        if (w < w_searched) { return true; }
        weights[w]++;
        w = popcount(helper_GF2[0][2]);
        if (w < w_searched) { return true; }
        weights[w]++;
        w = popcount(helper_GF2[0][3]);
        if (w < w_searched) { return true; }
        weights[w]++;

        linear_combinations_64_256_less_than(1, 1);
    }
    else if (n < 128) {
        set_128_256(bits);
        unsigned int w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        if (w < w_searched) { return true; }
        weights[w]++;
        linear_combinations_128_256_less_than(1, 1);
    }
    else {
        set_256(bits);
        linear_combinations_256_less_than(1, 1);

    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}



bool calculateWeightCH2_256_less_than(dynamic_mat_short& bits, int n, int k, int m, int d) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    less_than_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    setMatrixGF2_CF_256(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_256_less_than(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_256_less_than(1, 1);
        }
        else {
            linear_combinations_CF2_256_less_than(1, 1);
        }
        break;
    case 8:
        linear_combinations_CF2_256_less_than(1, 1);
        break;
    case 16:
        linear_combinations_CF2_256_less_than(1, 1);
        break;
    case 32:
        linear_combinations_CF2_256_less_than(1, 1);
        break;
    case 64:
        linear_combinations_CF2_256_less_than(1, 1);
        break;
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}

bool calculateWeightCH3_256_less_than(dynamic_mat_short& bits, int n, int k, int m, int d) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    less_than_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 256) + 1);

    if (m == 1) {
        Q = 3;
        setMatrixGF3_256(bits);
        if (n <= 64) {

            linear_comb_recGF3_64_256_less_than(1, 1);
            // linear_comb_recGF3_SSE(1, 1);

        }
        else if (n <= 128) {
            linear_comb_recGF3_128_256_less_than(1, 1);
        }
        else {
            linear_comb_recGF3_AVX_less_than(1, 1);
        }


    }
    else if (m == 2) {
        Q = 9;
        setMatrixGF9_256(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_256_less_than(1, 1);

        }
        else if (n <= 128) {
            linear_combinations_GF9_128_256_less_than(1, 1);
        }
        else {
            linear_combinations_GF9_AVX_less_than(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
        setMatrixGF27_256(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_256_less_than(1, 1);

        }
        else if (n <= 128) {
            linear_combinations_GF27_128_256_less_than(1, 1);
        }
        else {
            linear_combinations_GF27_AVX_less_than(1, 1);
        }

    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}



//----------------------------------------------------Chaeck if there is a word we weight equal to  given w----------------------------------------------------------------//



bool calculateWeightCH3_256_equal(dynamic_mat_short& bits, int n, int k, int m, int d) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    equal_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 256) + 1);

    if (m == 1) {
        Q = 3;
        setMatrixGF3_256(bits);
        if (n <= 64) {
            linear_comb_recGF3_64_256_equal(1, 1);
        }
        else if (n <= 128) {
            linear_comb_recGF3_128_256_equal(1, 1);
        }
        else {
            linear_comb_recGF3_AVX_equal(1, 1);
        }


    }
    else if (m == 2) {
        Q = 9;
        setMatrixGF9_256(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_256_equal(1, 1);

        }
        else if (n <= 128) {
            linear_combinations_GF9_128_256_equal(1, 1);
        }
        else {
            linear_combinations_GF9_AVX_equal(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
        setMatrixGF27_256(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_256_equal(1, 1);

        }
        else if (n <= 128) {
            linear_combinations_GF27_128_256_equal(1, 1);

        }
        else {
            linear_combinations_GF27_AVX_equal(1, 1);
        }

    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}




bool calculateWeightCH2_256_equal(dynamic_mat_short& bits, int n, int k, int m, int d) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    equal_flag = true;


    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    setMatrixGF2_CF_256(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_256_equal(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_256_equal(1, 1);
        }
        else {
            linear_combinations_CF2_256_equal(1, 1);
        }
        break;
    case 8:
        linear_combinations_CF2_256_equal(1, 1);
        break;
    case 16:
        linear_combinations_CF2_256_equal(1, 1);
        break;
    case 32:
        linear_combinations_CF2_256_equal(1, 1);
        break;
    case 64:
        linear_combinations_CF2_256_equal(1, 1);
        break;
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}





bool calculateWeightGF2_256_equal(dynamic_mat_short& bits, int n, int k, int d) {
    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    equal_flag = true;

    register_elements = ((n - 1) / 256) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (n <= 64) {
        set_64_256(bits);
        unsigned long long int w = popcount(helper_GF2[0][1]);
        if (w == w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][2]);
        if (w == w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][3]);
        if (w == w_searched) return true;
        weights[w]++;
        linear_combinations_64_256_equal(1, 1);
    }
    else if (n <= 128) {
        set_128_256(bits);

        unsigned long long int w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        if (w == w_searched) return true;
        weights[w]++;
        linear_combinations_128_256_equal(1, 1);
    }
    else {
        set_256(bits);
        linear_combinations_256_equal(1, 1);

    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}




bool calculateWeightBytes_256_equal(dmat_type& bits, int n, int k, int m, int q, int d) {
    
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    equal_flag = true;
    register_elements = ((n - 1) / 256) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes_256(bits);
            linear_combinations_CH2_256_equal(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersCF_256(bits);
            linear_combinations_CF_49_256_equal(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersCF_256(bits);
            linear_combinations_CF_25_256_equal(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes_256(bits);
        linear_combinations_Bytes_256_equal(1, 1);
    }


    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}




//----------------------------------------------------END - Chaeck if there is a word we weight equal to  given w----------------------------------------------------------------//



//---------------------------------------Calculate the number of codewords with given weight -------------------------------------------------//

unsigned long long int calculateNumberOfWordsGF2_256_equal(dynamic_mat_short& bits, int n, int k, int d, bool multiplicativeForm) {
    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    form = multiplicativeForm;


    file = fopen("Result_codewords_CountEqual.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight = %d:\n", w_searched);
    }
    else {
        printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
    }
    register_elements = ((n - 1) / 256) + 1;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (n <= 64) {
        set_64_256(bits);
        unsigned int w = popcount(helper_GF2[0][1]);
        weights[w]++;
        if (w == w_searched) {
            write_GF2_coset_64(0, 1);
        }

        w = popcount(helper_GF2[0][2]);
        weights[w]++;
        if (w == w_searched) {
            write_GF2_coset_64(0, 2);
        }
        w = popcount(helper_GF2[0][3]);
        weights[w]++;
        if (w == w_searched) {
            write_GF2_coset_64(0, 3);
        }

        linear_combinations_64_256_equal_count(1, 1);
    }
    else if (n <= 128) {
        set_128_256(bits);

        unsigned long long int w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        weights[w]++;
        if (w == w_searched) {
            write_GF2_coset_256(0, 2);
            //write_GF2_coset(0, 3);
        }

        linear_combinations_128_256_equal_count(1, 1);
    }
    else {
        set_256(bits);
        linear_combinations_256_equal_count(1, 1);

    }

    unsigned long long int ct = weights[w_searched];
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }

    return ct;
}





unsigned long long int  calculateNumberOfWordsBytes_256_equal(dmat_type& bits, int n, int k, int m, int q, int d, bool multiplicativeForm) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords_CountEqual.txt", "a");


    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight =  %d:\n", w_searched);
            }
            else {
                printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
            }
            setRegistersBytes_256(bits);
            linear_combinations_CH2_256_equal_count(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight =  %d:\n", w_searched);
            }
            else {
                printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
            }
            setRegistersCF_256(bits);
            linear_combinations_CF_49_256_equal_count(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight =  %d:\n", w_searched);
            }
            else {
                printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
            }
            setRegistersCF_256(bits);
            linear_combinations_CF_25_256_equal_count(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        if (file != NULL) {
            fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
            fprintf(file, "Searching for words with weight =  %d:\n", w_searched);
        }
        else {
            printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
        }
        setRegistersBytes_256(bits);
        linear_combinations_Bytes_256_euqal_count(1, 1);
    }

    unsigned long long int ct = weights[w_searched];
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }

    return ct;
}

unsigned long long int calculateNumberOfWordsCH2_256_equal(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords_CountEqual.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight = %d:\n", w_searched);
    }
    else {
        printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    setMatrixGF2_CF_256(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_256_equal_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_256_equal_count(1, 1);
        }
        else {
            linear_combinations_CF2_256_equal_count(1, 1);
        }
        break;
    case 8:
        linear_combinations_CF2_256_equal_count(1, 1);
        break;
    case 16:
        linear_combinations_CF2_256_equal_count(1, 1);
        break;
    case 32:
        linear_combinations_CF2_256_equal_count(1, 1);
        break;
    case 64:
        linear_combinations_CF2_256_equal_count(1, 1);
        break;
    }
    unsigned long long int ct = weights[w_searched];
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }

    return ct;
}

unsigned long long int calculateNumberOfWordsCH3_256_equal(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    form = multiplicativeForm;

    Q = 1;
    for (int i = 0; i < m; i++) {
        Q = Q * 3;
    }

    file = fopen("Result_codewords_CountEqual.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight = %d:\n", w_searched);
    }
    else {
        printf("Cannot open file Result_codewords_CountEqual.txt\n The codewords won't be writen!\n");
    }


    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 256) + 1);

    if (m == 1) {
        Q = 3;
        setMatrixGF3_256(bits);
        if (n <= 64) {

            linear_comb_recGF3_64_256_equal_count(1, 1);
            // linear_comb_recGF3_SSE(1, 1);

        }else if (n <= 128) {
            linear_comb_recGF3_128_256_equal_count(1, 1);
        }
        else {
            linear_comb_recGF3_AVX_equal_count(1, 1);
        }

    }
    else if (m == 2) {
        Q = 9;
        setMatrixGF9_256(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_256_equal_count(1, 1);

        }else if (n <= 128) {
            linear_combinations_GF9_128_256_equal_count(1, 1);

        }
        else {
            linear_combinations_GF9_AVX_equal_count(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
        setMatrixGF27_256(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_256_equal_count(1, 1);

        }
        else if (n <= 128) {
            linear_combinations_GF27_128_256_equal_count(1, 1);

        }
        else {
            linear_combinations_GF27_AVX_equal_count(1, 1);
        }

    }


    unsigned long long int ct = weights[w_searched];
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }

    return ct;

}



//------------------------------------------------------ END  -----------------------------------------------------------------//


//---------------------------------------------------- Calculate the number of words with weight less than given value --------------------------------------------------------//



unsigned long long int calculateNumberOfWordsGF2_256_less_than(dynamic_mat_short& bits, int n, int k, int d, bool multiplicativeForm) {
    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;

    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords_CountLessThan.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight < %d:\n", w_searched);
    }
    else {
        printf( "Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
    }

    register_elements = ((n - 1) / 256) + 1;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (n <= 64) {
        set_64_256(bits);
        unsigned long int w = popcount(helper_GF2[0][1]);
        weights[w]++;
        if (w < w_searched) {
            write_GF2_coset_64(0, 1);
        }
        w = popcount(helper_GF2[0][2]);
        weights[w]++;
        if (w < w_searched) {
            write_GF2_coset_64(0, 2);
        }
        w = popcount(helper_GF2[0][3]);
        weights[w]++;
        if (w < w_searched) {
            write_GF2_coset_64(0, 3);
        }
        linear_combinations_64_256_less_than_count(1, 1);
    }
    else if (n <= 128) {
        set_128_256(bits);
        unsigned long int w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        weights[w]++;
        if (w < w_searched) {
            write_GF2_coset_256(0, 2);
        }
        linear_combinations_128_256_less_than_count(1, 1);
    }
    else {
        set_256(bits);
        linear_combinations_256_less_than_count(1, 1);

    }
    unsigned long long int ct = 0;
    unsigned long long int i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }
    return ct;
}



unsigned long long int  calculateNumberOfWordsBytes_256_less_than(dmat_type& bits, int n, int k, int m, int q, int d, bool multiplicativeForm) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;

    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords_CountLessThan.txt", "a");

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight < %d:\n", w_searched);
            }
            else {
                printf( "Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
            }
            setRegistersBytes_256(bits);
            linear_combinations_CH2_256_less_than_count(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight < %d:\n", w_searched);
            }
            else {
                printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
            }
            setRegistersCF_256(bits);
            linear_combinations_CF_49_256_less_than_count(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight < %d:\n", w_searched);
            }
            else {
                printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
            }
            setRegistersCF_256(bits);
            linear_combinations_CF_25_256_less_than_count(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        if (file != NULL) {
            fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
            fprintf(file, "Searching for words with weight < %d:\n", w_searched);
        }
        else {
            printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
        }
        setRegistersBytes_256(bits);
        linear_combinations_Bytes_256_less_than_count(1, 1);
    }
    unsigned long long int ct = 0;
    unsigned long long int i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }
    return ct;
}






unsigned long long int calculateNumberOfWordsCH2_256_less_than(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords_CountLessThan.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight < %d:\n", w_searched);
    }
    else {
        printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
    }
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }



    setMatrixGF2_CF_256(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_256_less_than_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_256_less_than_count(1, 1);
        }
        else {
            linear_combinations_CF2_256_less_than_count(1, 1);
        }
        break;
    case 8:
        linear_combinations_CF2_256_less_than_count(1, 1);
        break;
    case 16:
        linear_combinations_CF2_256_less_than_count(1, 1);
        break;
    case 32:
        linear_combinations_CF2_256_less_than_count(1, 1);
        break;
    case 64:
        linear_combinations_CF2_256_less_than_count(1, 1);
        break;
    }


    unsigned long long int ct = 0;
    unsigned long long int i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }
    return ct;
}



unsigned long long int calculateNumberOfWordsCH3_256_less_than(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {

    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    form = multiplicativeForm;
    Q = 1;
    for (int i = 0; i < m; i++) {
        Q = Q * 3;
    }
    file = fopen("Result_codewords_CountLessThan.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight < %d:\n", w_searched);
    }
    else {
       printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 256) + 1);

    if (m == 1) {
        //  Q = 3;
        setMatrixGF3_256(bits);
        if (n <= 64) {
            linear_comb_recGF3_64_256_less_than_count(1, 1);
        }else if (n <= 128) {
            linear_comb_recGF3_128_256_less_than_count(1, 1);
        }
        else {
            linear_comb_recGF3_AVX_less_than_count(1, 1);
        }


    }
    else if (m == 2) {
        setMatrixGF9_256(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_256_less_than_count(1, 1);
        }else if (n <= 128) {
            linear_combinations_GF9_128_256_less_than_count(1, 1);
        }
        else {
            linear_combinations_GF9_AVX_less_than_count(1, 1);
        }
    }
    else if (m == 3) {
        setMatrixGF27_256(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_256_less_than_count(1, 1);
        }else if (n <= 128) {
            linear_combinations_GF27_128_256_less_than_count(1, 1);
        }
        else {
            linear_combinations_GF27_AVX_less_than_count(1, 1);
        }

    }

    unsigned long long int ct = 0;
    unsigned long long int  i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    if (file != NULL) {
        fprintf(file, "\n\n");
        fclose(file);
    }
    return ct;

}
#else

void calculateWeightCH3_256(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m) { 
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory."); 
}
void calculateWeightBytes_256(dmat_type& generatorMatrix_byte, int n, int k, int m, int q) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
}
void calculateWeightCH2_256(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
}
void calculateWeightGF2_256(dynamic_mat_short& generatorMatrix_bits, int n, int k) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
}
bool calculateWeightCH3_256_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightBytes_256_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightCH2_256_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightGF2_256_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightCH3_256_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int d) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightBytes_256_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int d) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightCH2_256_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int d) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightGF2_256_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int d) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
unsigned long long int calculateNumberOfWordsCH3_256_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsBytes_256_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsCH2_256_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsGF2_256_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsCH3_256_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsBytes_256_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsCH2_256_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsGF2_256_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX2 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX2, or manually change the flags for project v1.3! Otherwise, please add the flag -mavx2 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}

#endif