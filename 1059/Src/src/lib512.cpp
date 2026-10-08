#include <iostream>
#include <fstream>
#include "lib512.h"


#if defined(_MSC_VER) || (__INTEL_LLVM_COMPILER)
#include <intrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <dvec.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#include <cpuid.h>
#endif

using namespace std;

#if defined (__AVX512F__ )&&(__AVX512BW__)

// static arrays that are used to save the generator matrix into the registers

// static arrays that are used to save the generator matrix into the registers
static union {
    __m128i reg128_matrix_GF2[K_GF2][N_GF2 / 2];
    __m256i reg256_matrix_GF2[K_GF2][N_GF2 / 4];
    __m512i reg512_matrix_GF2[K_GF2][N_GF2 / 8];
    unsigned long long int matrix_GF2[K_GF2][N_GF2]; // for GF2 and GF4; bitwise representation of the elements of the field with characteristic 2
    __m128i reg128_matrix_CH2[K_GF2][N_CH2 / 16];
    __m256i reg256_matrix_CH2[K_GF2][N_CH2 / 32];
    __m512i reg512_matrix_CH2[K_GF2][N_CH2 / 64];
    unsigned char matrix_CH2[K_GF2][N_CH2]; // for GF8, GF16, GF32, GF64; bytewise representation of the elements ofthe field with characteristic 2
    unsigned long long int matrix_CH3[K_CH3][N_CH3]; // for fields with characteristic 3; bitwise representation of the elements
    __m128i reg128_matrix_CH3[K_CH3][N_CH3 / 2];
    __m256i reg256_matrix_CH3[K_CH3][N_CH3 / 4];
    __m512i reg512_matrix_CH3[K_CH3][N_CH3 / 8];
    unsigned char matrix_p[K_P][N_P]; // for ohter finite fields (GF5, GF7, GF11, ..., GF25, ..., GF49, ...); bytewise representation of the elements
    __m128i reg128_matrix_p[K_P][N_P / 16];
    __m256i reg256_matrix_p[K_P][N_P / 32];
    __m512i reg512_matrix_p[K_P][N_P / 64];
};

// array that save the current linear combination
// used to calculate naext linear combination
static union {
    __m512i reg512_helper_GF2[K_GF2][N_GF2 / 8];
    __m256i reg256_helper_GF2[K_GF2][N_GF2 / 4];
    __m128i reg128_helper_GF2[K_GF2][N_GF2 / 2];
    unsigned long long int helper_GF2[K_GF2][N_GF2];
    unsigned char helper_CH2[K_GF2][N_CH2];
    __m512i reg512_helper_CH2[K_GF2][N_CH2 / 64];
    __m256i reg256_helper_CH2[K_GF2][N_CH2 / 32];
    __m128i reg128_helper_CH2[K_GF2][N_CH2 / 16];
    unsigned long long int helper_CH3[K_CH3][N_CH3];
    __m512i reg512_helper_CH3[K_CH3][N_CH3 / 8];
    __m256i reg256_helper_CH3[K_CH3][N_CH3 / 4];
    __m128i reg128_helper_CH3[K_CH3][N_CH3 / 2];
    unsigned char helper_p[K_P][N_P];
    __m512i reg512_helper_p[K_P][N_P / 64];
    __m256i reg256_helper_p[K_P][N_P / 32];
    __m128i reg128_helper_p[K_P][N_P / 16];
};

static __m512i zero;
static __m512i Q_reg_Bytes;

//-------------------popcnt instr functions --------------------------------//

static inline unsigned long long int weight_P64(__m512i& reg) {
#if (__AVX512VPOPCNTDQ__) || (AVX512POCNT_MANUAL==1)
    //printf("popcnt644\n");
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    unsigned long long int w = 0;
    w_r = _mm512_popcnt_epi64(reg);
    w = w64[0];
    return w;
#else
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    w_r = reg;
    return popcount(w64[0]);
#endif
}

static inline unsigned long long int weight_P128(__m512i& reg) {
#if (__AVX512VPOPCNTDQ__) || (AVX512POCNT_MANUAL==1)
   // printf("popcnt128\n");
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    unsigned long long int w = 0;
    w_r = _mm512_popcnt_epi64(reg);
    w = w64[0] + w64[1];
    return w;
#else
static union {
    __m512i w_r;
    unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
};
w_r = reg;
return popcount(w64[0]) + popcount(w64[1]);
#endif
}

static inline unsigned long long int weight_P256(__m512i& reg) {
#if (__AVX512VPOPCNTDQ__) || (AVX512POCNT_MANUAL==1)
    //printf("popcnt256\n");
    __m512i w_r = _mm512_setzero_si512();
    unsigned long long int w = 0;
    w_r = _mm512_popcnt_epi64(reg);
    w = _mm512_reduce_add_epi64(w_r) >> 1;
    return w;
#else
static union {
    __m512i w_r;
    unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
};
w_r = reg;
return popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]);
#endif
}

static inline unsigned long long int weight_red_add(__m512i& reg) {
#if (__AVX512VPOPCNTDQ__) || (AVX512POCNT_MANUAL==1)
    //printf("popcnt512\n");
    unsigned long long int w = 0; __m512i weight = _mm512_setzero_si512();
    weight = _mm512_popcnt_epi64(reg);
    w = w + _mm512_reduce_add_epi64(weight);
    return w;
#else
static union {
    __m512i w_r;
    unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
};
w_r = reg;
return popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]) +
    popcount(w64[4]) + popcount(w64[5]) + popcount(w64[6]) + popcount(w64[7]);
#endif
}
//-------------------popcnt instr functions --------------------------------//


//----------------------------- additional writing functions for 512-bit registers -------------------------//

void write_GF2_512(int res) {
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

void write_GF2_coset_64_512(int res, int el) {
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

void write_GF2_coset_128_512(int res, int el) {
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
            if ((1) * 64 + shift > (N - 1)) {
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

void write_GF2_coset_512(int res, int el) {
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
            if (1 * 64 + shift > (N - 1)) {
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

        for (int shift = 0; shift < 64; shift++) {
            if (2 * 64 + shift > (N - 1)) {
                fprintf(file, "\n");
                return;
            }
            if (helper_GF2[res][el + 2] & (one << (63 - shift))) {
                fprintf(file, "%d", 1);
            }
            else {
                fprintf(file, "%d", 0);
            }
        }

        for (int shift = 0; shift < 64; shift++) {
            if (3 * 64 + shift > (N - 1)) {
                fprintf(file, "\n");
                return;
            }
            if (helper_GF2[res][el + 3] & (one << (63 - shift))) {
                fprintf(file, "%d", 1);
            }
            else {
                fprintf(file, "%d", 0);
            }

        }

        fprintf(file, "\n");
    }
}


void write_ByteCH2_512(int res) {
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
void write_CF_512(int res) {
    if (file != NULL) {
        int t = 0;
        int ch = 5;
        if (Q == 49) ch = 7;
        int shift = (((N - 1) / 64) + 1);
        for (int i = 0; i < N; i++) {
            t = 0;
            t = (ch * helper_p[res][i]) + helper_p[res][i + 64 * shift];
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

void write_Bytes_512(int res) {
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


void write_GF3_512(int res) {
    if (file != NULL) {
        int c = (((N - 1) / 64) + 1);
        int bit1 = 0;
        if (c < 3 || c == 4) {
            bit1 = c;
        }
        else if (c == 3) {
            bit1 = 4;
        }
        else {
            bit1 = 8 * register_elements;
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
                first = helper_CH3[res][i] & (one << (63 - shift));
                second = helper_CH3[res][i + bit1] & (one << (63 - shift));

                if (first && second) { fprintf(file, "%d", 0); }
                else if (first) { fprintf(file, "%d", 1); }
                else if (second) { fprintf(file, "%d", 2); }
                else { printf("ERROR in writing in file for GF3 - element is 00!\n\n\n"); return; }
            }
        }
        fprintf(file, "\n");
    }
}

void write_GF9_512(int res) {
    if (file != NULL) {
        int c = (((N - 1) / 64) + 1);

        int bit1 = 0;
        if (c < 3 || c == 4) {
            bit1 = c;
        }
        else if (c == 3) {
            bit1 = 4;
        }
        else {
            bit1 = 8 * register_elements;
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
                first = helper_CH3[res][i] & (one << (63 - shift));
                second = helper_CH3[res][i + bit1] & (one << (63 - shift));


                if (first != 0 && second != 0) { temp = 0; }
                else if (first != 0) { temp = 1; }
                else if (second != 0) { temp = 2; }
                else { printf("EROR in writing in file for GF9 - element is 00!\n\n\n"); return; }
                result = result + temp;


                first = helper_CH3[res][i + 2 * bit1] & (one << (63 - shift));
                second = helper_CH3[res][i + 3 * bit1] & (one << (63 - shift));

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

void write_GF27_512(int res) {
    if (file != NULL) {
        int c = (((N - 1) / 64) + 1);

        int bit1 = 0;
        if (c <= 4) {
            bit1 = 4;
        }
        else {
            bit1 = 8 * register_elements;
        }
        unsigned long long int one = 1;


        int temp = 0;
        int result = 0;
        bool first, second;

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
                else { printf("EROR in witing in file for GF27 - element is 00!\n\n\n"); return; }
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



        fprintf(file, "\n");
    }
}

void write_CF2_512(int res) {
    if (file != NULL) {
        unsigned long long int one = 1;
        int c = ((N - 1) / 64) + 1;
        int bit1 = 0;
        if (N < 256) {
            bit1 = 4;
        }
        else {
            bit1 = 8 * register_elements;
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



//-----------------------Functions Bytes ------------------------------------//
void setRegistersBytes_512(dmat_type& bits) {
    // printf( "Set registers\n");
    zero = _mm512_setzero_si512();
    Q_reg_Bytes = _mm512_set1_epi8((char)Q);
    //Q_reg_Bytes = _mm512_set_epi8((char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q,
    //    (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q,
    //    (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q,
    //    (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q,
    //    (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q, (char)Q);

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

            }
        }
    }

}


void setRegistersBytesCF_512(dmat_type& bits) {
    zero = _mm512_setzero_si512();
    Q_reg_Bytes = _mm512_set1_epi8((char)Characteristic);
    //Q_reg_Bytes = _mm512_set_epi8((char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic,
   //     (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic,
    //    (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic,
    //    (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic,
    //    (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic, (char)Characteristic);
    for (int row = 0; row < K_P; row++) {
        for (int col = 0; col < N_P; col++) {
            matrix_p[row][col] = 0;
            helper_p[row][col] = 0;
        }
    }
    int shift = (((N - 1) / 64) + 1);
    for (int row = 1; row <= (M * K); row++) {
        for (int col = 0; col < N; col++) {
            matrix_p[row][col] = bits.a[row - 1][col];
            matrix_p[row][col + 64 * shift] = bits.a[row - 1][col + N];
        }
    }
}


static inline void addBytes_512(int rec, int i, int res) {
    __m512i res_add, res_sub;
    __mmask64  mask;
    unsigned long long int w = 0;
    for (int col = 0; col < (((N - 1) / 64) + 1); col++) {
        res_add = _mm512_add_epi8(reg512_helper_p[rec][col], reg512_matrix_p[i][col]);
        res_sub = _mm512_sub_epi8(res_add, Q_reg_Bytes);
        mask = _mm512_cmplt_epi8_mask(res_sub, zero);
        reg512_helper_p[res][col] = _mm512_mask_blend_epi8(mask, res_sub, res_add);
    }
}

static inline void addBytes_CF_512(int rec, int i, int res) {
    __m512i res_add, res_sub;
    __mmask64  mask;
    unsigned long long int w = 0;
    int shift = (((N - 1) / 64) + 1);
    for (int col = 0; col < shift; col++) {
        // ^0
        res_add = _mm512_add_epi8(reg512_helper_p[rec][col], reg512_matrix_p[i][col]);
        res_sub = _mm512_sub_epi8(res_add, Q_reg_Bytes);
        mask = _mm512_cmplt_epi8_mask(res_sub, zero);
        reg512_helper_p[res][col] = _mm512_mask_blend_epi8(mask, res_sub, res_add);

        // ^1
        res_add = _mm512_add_epi8(reg512_helper_p[rec][col + shift], reg512_matrix_p[i][col + shift]);
        res_sub = _mm512_sub_epi8(res_add, Q_reg_Bytes);
        mask = _mm512_cmplt_epi8_mask(res_sub, zero);
        reg512_helper_p[res][col + shift] = _mm512_mask_blend_epi8(mask, res_sub, res_add);
    }
}

static inline void add_CH2_512(int rec, int i, int res) {
    for (int col = 0; col < ((N - 1) / 64) + 1; col++) {
        reg512_helper_CH2[res][col] = _mm512_xor_si512(reg512_helper_CH2[rec][col], reg512_matrix_CH2[i][col]);
    }
}

static inline unsigned long long int weightBytes_CF_512(int res) {
    unsigned long long int w = 0;
    int shift = (((N - 1) / 64) + 1);
    __mmask64  mask1, mask2, mask;
    for (int col = 0; col < (((N - 1) / 64) + 1); col++) {
        mask1 = _mm512_cmpeq_epi8_mask(reg512_helper_p[res][col], zero);
        mask2 = _mm512_cmpeq_epi8_mask(reg512_helper_p[res][col + shift], zero);
        mask = _kand_mask64(mask1, mask2);
        //  mask = mask1 & mask2;
        w = w + (64 - popcount(mask));
    }

    return w;
}

static inline unsigned long long int weightBytes_512(int res) {
    unsigned long long int w = 0;
    for (int col = 0; col < (((N - 1) / 64) + 1); col++) {
        w = w + (64 - popcount(_mm512_cmpeq_epi8_mask(reg512_helper_p[res][col], zero)));
    }
    return w;
}

static inline unsigned long long int weightBytes_CH2_512(int res) {
    unsigned long long int w = 0;
    for (int col = 0; col < (((N - 1) / 64) + 1); col++) {
        w = w + (64 - popcount(_mm512_cmpeq_epi8_mask(reg512_helper_CH2[res][col], zero)));
    }
    return w;
}

void linear_combinations_Bytes_512(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_512(rec - 1, i, rec);
                unsigned long long int w = weightBytes_512(rec);
                weights[w]++;
            }
            else {
                addBytes_512(rec, i, rec);
                unsigned long long int w = weightBytes_512(rec);
                weights[w]++;
            }

            if (rec < K) {
                linear_combinations_Bytes_512(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CF_49_512(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_CF_512(rec - 1, i, rec);
                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
            }
            else {
                short t = TransitionSequence49[q1] - 1;
                addBytes_CF_512(rec, t * K + i, rec);
                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
            }
            if (rec < K) {
                linear_combinations_CF_49_512(rec + 1, i + 1);
            }
        }
    }
}


void linear_combinations_CF_25_512(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_CF_512(rec - 1, i, rec);
                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
            }
            else {
                short t = TransitionSequence25[q1] - 1;
                addBytes_CF_512(rec, t * K + i, rec);
                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
            }
            if (rec < K) {
                linear_combinations_CF_25_512(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_CH2_512(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            unsigned long long int w = 0;
            if (q1 == 1) {
                add_CH2_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CH2_512(rec, t * K + i, rec);
            }

            w = weightBytes_CH2_512(rec);
            weights[w]++;

            if (rec < K) {
                linear_combinations_CH2_512(rec + 1, i + 1);
            }
        }
    }
}

//---equal--//

void linear_combinations_Bytes_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    addBytes_512(rec - 1, i, rec);
                }
                else { addBytes_512(rec, i, rec); }

                unsigned long long int w = weightBytes_512(rec);
                weights[w]++;
                if (w == w_searched) { equal_flag = false; break; }

                if (rec < K) {
                    linear_combinations_Bytes_512_equal(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_Bytes_512_equal_count(int rec, int h) {

    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_512(rec - 1, i, rec);
            }
            else { addBytes_512(rec, i, rec); }

            unsigned long long int w = weightBytes_512(rec);
            weights[w]++;
            if (w == w_searched) { write_Bytes_512(rec); }

            if (rec < K) {
                linear_combinations_Bytes_512_equal_count(rec + 1, i + 1);
            }
        }
    }


}

void linear_combinations_Bytes_512_less_count(int rec, int h) {

    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_512(rec - 1, i, rec);
            }
            else { addBytes_512(rec, i, rec); }

            unsigned long long int w = weightBytes_512(rec);
            weights[w]++;
            if (w < w_searched) { write_Bytes_512(rec); }

            if (rec < K) {
                linear_combinations_Bytes_512_less_count(rec + 1, i + 1);
            }
        }
    }


}

void linear_combinations_CF_49_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    addBytes_CF_512(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    addBytes_CF_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
                if (w == w_searched) { equal_flag = false; break; }
                if (rec < K) {
                    linear_combinations_CF_49_512_equal(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_CF_49_512_equal_count(int rec, int h) {

    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_CF_512(rec - 1, i, rec);
            }
            else {
                short t = TransitionSequence49[q1] - 1;
                addBytes_CF_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weightBytes_CF_512(rec);
            weights[w]++;
            if (w == w_searched) { write_CF_512(rec); }
            if (rec < K) {
                linear_combinations_CF_49_512_equal_count(rec + 1, i + 1);
            }
        }
    }

}

void linear_combinations_CF_49_512_less_count(int rec, int h) {

    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_CF_512(rec - 1, i, rec);
            }
            else {
                short t = TransitionSequence49[q1] - 1;
                addBytes_CF_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weightBytes_CF_512(rec);
            weights[w]++;
            if (w < w_searched) { write_CF_512(rec); }
            if (rec < K) {
                linear_combinations_CF_49_512_less_count(rec + 1, i + 1);
            }
        }
    }

}

void linear_combinations_CF_25_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    addBytes_CF_512(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    addBytes_CF_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
                if (w == w_searched) { equal_flag = false; break; }
                if (rec < K) {
                    linear_combinations_CF_25_512_equal(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_CF_25_512_equal_count(int rec, int h) {

    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_CF_512(rec - 1, i, rec);
            }
            else {
                short t = TransitionSequence25[q1] - 1;
                addBytes_CF_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weightBytes_CF_512(rec);
            weights[w]++;
            if (w == w_searched) { write_CF_512(rec); }
            if (rec < K) {
                linear_combinations_CF_25_512_equal_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_CF_25_512_less_count(int rec, int h) {

    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                addBytes_CF_512(rec - 1, i, rec);
            }
            else {
                short t = TransitionSequence25[q1] - 1;
                addBytes_CF_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weightBytes_CF_512(rec);
            weights[w]++;
            if (w < w_searched) { write_CF_512(rec); }
            if (rec < K) {
                linear_combinations_CF_25_512_less_count(rec + 1, i + 1);
            }
        }
    }


}


void linear_combinations_CH2_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2_512(rec, t * K + i, rec);
                }
                unsigned long long int w = 0;
                w = weightBytes_CH2_512(rec);
                weights[w]++;
                if (w == w_searched) { equal_flag = false; break; }
                if (rec < K) {
                    linear_combinations_CH2_512_equal(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_CH2_512_equal_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CH2_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CH2_512(rec, t * K + i, rec);
            }
            unsigned long long int weight = 0;
            weight = weightBytes_CH2_512(rec);
            weights[weight]++;
            if (weight == w_searched) { write_ByteCH2_512(rec); }
            if (rec < K) {
                linear_combinations_CH2_512_equal_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_CH2_512_less_count(int rec, int h) {
    int qf = Q;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if (q1 == 1) {
                add_CH2_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_CH2_512(rec, t * K + i, rec);
            }
            unsigned long long int weight = 0;
            weight = weightBytes_CH2_512(rec);
            weights[weight]++;
            if (weight < w_searched) { write_ByteCH2_512(rec); }
            if (rec < K) {
                linear_combinations_CH2_512_less_count(rec + 1, i + 1);
            }
        }
    }
}

//---less than---//

void linear_combinations_Bytes_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    addBytes_512(rec - 1, i, rec);
                }
                else { addBytes_512(rec, i, rec); }

                unsigned long long int w = weightBytes_512(rec);
                weights[w]++;
                if (w < w_searched) { less_than_flag = false; break; }

                if (rec < K) {
                    linear_combinations_Bytes_512_less(rec + 1, i + 1);
                }
            }
        }
    }

}


void linear_combinations_CF_49_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    addBytes_CF_512(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    addBytes_CF_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
                if (w < w_searched) { less_than_flag = false; break; }
                if (rec < K) {
                    linear_combinations_CF_49_512_less(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_CF_25_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    addBytes_CF_512(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    addBytes_CF_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weightBytes_CF_512(rec);
                weights[w]++;
                if (w < w_searched) { less_than_flag = false; break; }
                if (rec < K) {
                    linear_combinations_CF_25_512_less(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_CH2_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2_512(rec, t * K + i, rec);
                }
                unsigned long long int weight = 0;
                weight = weightBytes_CH2_512(rec);
                weights[weight]++;
                if (weight < w_searched) { less_than_flag = false; break; }
                if (rec < K) {
                    linear_combinations_CH2_512_less(rec + 1, i + 1);
                }
            }
        }
    }

}

//-----------------------Functions Bytes ------------------------------------//

//------------------------------------- Functions GF2  -----------------------------------------//

void set_64_512(dynamic_mat_short& bits) {
    //   |-----c-----|---c+g_k---|-c+g_(k-1)-|c+g_k+g_(k-1)|-----64----|-----64----|-----64----|-----64----|
    //   |-----64----|-----64----|-----64----|------64-----|-----64----|-----64----|-----64----|-----64----|

    register_elements = (((N - 1) / 512) + 1);
    // cout << "Set set_64_512" << endl;
    reg512_matrix_GF2[0][0] = _mm512_setzero_si512();
    reg512_helper_GF2[0][0] = _mm512_setzero_si512();

    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        reg512_matrix_GF2[row][0] = _mm512_setzero_si512();
        reg512_helper_GF2[row][0] = _mm512_setzero_si512();


        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][0];
        matrix_GF2[row][2] = bits.a[row - 1][0];
        matrix_GF2[row][3] = bits.a[row - 1][0];

    }
    helper_GF2[0][1] = bits.a[K - 1][0];
    helper_GF2[0][2] = bits.a[K - 2][0];
    helper_GF2[0][3] = bits.a[K - 1][0] ^ bits.a[K - 2][0];

}

void set_128_512(dynamic_mat_short& bits) {
    //   |----------c------------|----------c+g_k----------|--------c+g_(k-1)------|-----c+g_k+g_(k-1)-----|
    //   |-----64----|-----64----|-----64----|------64-----|-----64----|-----64----|-----64----|-----64----|

    register_elements = (((N - 1) / 512) + 1);
    // cout << "Set set_128_512" << endl;
    reg512_matrix_GF2[0][0] = _mm512_setzero_si512();
    reg512_helper_GF2[0][0] = _mm512_setzero_si512();

    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        reg512_matrix_GF2[row][0] = _mm512_setzero_si512();
        reg512_helper_GF2[row][0] = _mm512_setzero_si512();

        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][1];

        matrix_GF2[row][2] = bits.a[row - 1][0];
        matrix_GF2[row][3] = bits.a[row - 1][1];

        matrix_GF2[row][4] = bits.a[row - 1][0];
        matrix_GF2[row][5] = bits.a[row - 1][1];

        matrix_GF2[row][6] = bits.a[row - 1][0];
        matrix_GF2[row][7] = bits.a[row - 1][1];

    }
    helper_GF2[0][2] = bits.a[K - 1][0];
    helper_GF2[0][3] = bits.a[K - 1][1];

    helper_GF2[0][4] = bits.a[K - 2][0];
    helper_GF2[0][5] = bits.a[K - 2][1];

    helper_GF2[0][6] = bits.a[K - 2][0] ^ bits.a[K - 1][0];
    helper_GF2[0][7] = bits.a[K - 2][1] ^ bits.a[K - 1][1];
}

void set_256_512(dynamic_mat_short& bits) {
    //   |-----------------------c-------------------------|---------------------c+g_k---------------------|
    //   |-----64----|-----64----|-----64----|------64-----|-----64----|-----64----|-----64----|-----64----|

    register_elements = (((N - 1) / 512) + 1);
    // cout << "Set registers cosets (256_512)" << endl;
    reg512_matrix_GF2[0][0] = _mm512_setzero_si512();
    reg512_helper_GF2[0][0] = _mm512_setzero_si512();

    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        reg512_matrix_GF2[row][0] = _mm512_setzero_si512();
        reg512_helper_GF2[row][0] = _mm512_setzero_si512();

        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][1];
        matrix_GF2[row][2] = bits.a[row - 1][2];
        matrix_GF2[row][3] = bits.a[row - 1][3];

        matrix_GF2[row][4] = bits.a[row - 1][0];
        matrix_GF2[row][5] = bits.a[row - 1][1];
        matrix_GF2[row][6] = bits.a[row - 1][2];
        matrix_GF2[row][7] = bits.a[row - 1][3];

    }
    helper_GF2[0][4] = bits.a[K - 1][0];
    helper_GF2[0][5] = bits.a[K - 1][1];
    helper_GF2[0][6] = bits.a[K - 1][2];
    helper_GF2[0][7] = bits.a[K - 1][3];


}

void set_512(dynamic_mat_short& bits) {
    //cout << "Set set_512" << endl;
    register_elements = (((N - 1) / 512) + 1);
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }
    for (int col = 0; col < register_elements; col++) {
        reg512_matrix_GF2[0][col] = _mm512_setzero_si512();
        reg512_helper_GF2[0][col] = _mm512_setzero_si512();
    }
    for (int row = 1; row <= K; row++) {
        for (int col = 0; col < (N_GF2 / 8); col++) {
            reg512_matrix_GF2[row][col] = _mm512_setzero_si512();
            reg512_helper_GF2[row][col] = _mm512_setzero_si512();
        }
        for (int el = 0; el < N_GF2; el++) {
            if ((el) * 64 > N) {
                break;
            }
            matrix_GF2[row][el] = bits.a[row - 1][el];
        }
    }
}

static inline void add_GF2_512(int rec, int i, int res) {
    unsigned long long int w = 0;
    // alignas(32)  unsigned long long int temp[4];
     //unsigned long long int* temp;
    for (int el = 0; el < register_elements; el++) {
        reg512_helper_GF2[res][el] = _mm512_xor_si512(reg512_helper_GF2[rec][el], reg512_matrix_GF2[i][el]);
    }

}

static inline unsigned long long int  weight_GF2_512(int res) {
    unsigned long long int w = 0; __m512i weight = _mm512_setzero_si512();
    for (int el = 0; el < register_elements; el++) {
       // weight = _mm512_popcnt_epi64(reg512_helper_GF2[res][el]);
        w = w + weight_red_add(reg512_helper_GF2[res][el]);//_mm512_reduce_add_epi64(weight);
    }
    return w;
}


static inline unsigned long long int  weight_GF2_512_f(int res) {
    unsigned long long int w = 0; __m512i weight = _mm512_setzero_si512();
    for (int el = 0; el < register_elements; el++) {
        w = w + popcount(helper_GF2[res][8 * el]) + popcount(helper_GF2[res][8 * el + 1])
            + popcount(helper_GF2[res][8 * el + 2]) + popcount(helper_GF2[res][8 * el + 3])
            + popcount(helper_GF2[res][8 * el + 4]) + popcount(helper_GF2[res][8 * el + 5])
            + popcount(helper_GF2[res][8 * el + 6]) + popcount(helper_GF2[res][8 * el + 7]);
    }
    return w;
}

void linear_combinationsGF2_64_512(int rec, int h) {
    for (int i = h; i < K - 1; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]);
        weights[w]++;

        w = popcount(helper_GF2[rec][1]);
        weights[w]++;

        w = popcount(helper_GF2[rec][2]);
        weights[w]++;

        w = popcount(helper_GF2[rec][3]);
        weights[w]++;

        if (rec < K - 2) { linear_combinationsGF2_64_512(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_128_512(int rec, int h) {
    for (int i = h; i < K - 1; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
        weights[w]++;

        w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        weights[w]++;

        w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]);
        weights[w]++;

        w = popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
        weights[w]++;

        if (rec < K - 2) { linear_combinationsGF2_128_512(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_256_512(int rec, int h) {
    for (int i = h; i < K; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]) + popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        weights[w]++;

        w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]) + popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
        weights[w]++;

        if (rec < K - 1) { linear_combinationsGF2_256_512(rec + 1, i + 1); }
    }
}

void linear_combinations_512(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_512(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_512(rec);
        weights[w]++;
        if (rec < K) {
            linear_combinations_512(rec + 1, j + 1);
        }
    }
}

void linear_combinations_512_f(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_512(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_512_f(rec);
        weights[w]++;
        if (rec < K) {
            linear_combinations_512_f(rec + 1, j + 1);
        }
    }
}

void linear_combinations_512_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_512(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_512(rec);
            if (w == w_searched) {
                equal_flag = false; break;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_equal(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_512_equal_count(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_512(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_512(rec);
        if (w == w_searched) {
            write_GF2_512(rec);
        }
        weights[w]++;
        if (rec < K) {
            linear_combinations_512_equal_count(rec + 1, j + 1);
        }
    }
}

void linear_combinations_512_less_count(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_512(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_512(rec);
        if (w < w_searched) {
            write_GF2_512(rec);
        }
        weights[w]++;
        if (rec < K) {
            linear_combinations_512_less_count(rec + 1, j + 1);
        }
    }
}

void linear_combinations_512_equal_f(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_512(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_512_f(rec);
            if (w == w_searched) {
                equal_flag = false; break;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_equal_f(rec + 1, j + 1);
            }
        }
    }
}


void linear_combinations_512_equal_count_f(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_512(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_512_f(rec);
        if (w == w_searched) {
            write_GF2_512(rec);
        }
        weights[w]++;
        if (rec < K) {
            linear_combinations_512_equal_count_f(rec + 1, j + 1);
        }
    }
}

void linear_combinations_512_less_count_f(int rec, int h) {
    for (int j = h; j <= K; j++) {
        add_GF2_512(rec - 1, j, rec);
        unsigned long long int w = weight_GF2_512_f(rec);
        if (w < w_searched) {
            write_GF2_512(rec);
        }
        weights[w]++;
        if (rec < K) {
            linear_combinations_512_less_count_f(rec + 1, j + 1);
        }
    }
}

void linear_combinationsGF2_256_512_equal(int rec, int h) {
    if (equal_flag) {
        for (int i = h; i < K; i++) {
            reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]) + popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
            if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]) + popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
            if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            if (rec < K - 1) { linear_combinationsGF2_256_512_equal(rec + 1, i + 1); }
        }
    }

}

void linear_combinationsGF2_256_512_equal_count(int rec, int h) {
    for (int i = h; i < K; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]) + popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        if (w == w_searched) { write_GF2_coset_512(rec, 0); }
        weights[w]++;

        w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]) + popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
        if (w == w_searched) { write_GF2_coset_512(rec, 4); }
        weights[w]++;

        if (rec < K - 1) { linear_combinationsGF2_256_512_equal_count(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_256_512_less_count(int rec, int h) {
    for (int i = h; i < K; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]) + popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        if (w < w_searched) { write_GF2_coset_512(rec, 0); }
        weights[w]++;

        w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]) + popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
        if (w < w_searched) { write_GF2_coset_512(rec, 4); }
        weights[w]++;

        if (rec < K - 1) { linear_combinationsGF2_256_512_less_count(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_128_512_equal(int rec, int h) {
    if (equal_flag) {
        for (int i = h; i < K - 1; i++) {
            reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
            if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
            if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]);
            if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
            if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            if (rec < K - 2) { linear_combinationsGF2_128_512_equal(rec + 1, i + 1); }
        }
    }
}

void linear_combinationsGF2_128_512_equal_count(int rec, int h) {
    for (int i = h; i < K - 1; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
        if (w == w_searched) { write_GF2_coset_128_512(rec, 0); }
        weights[w]++;

        w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        if (w == w_searched) { write_GF2_coset_128_512(rec, 2); }
        weights[w]++;

        w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]);
        if (w == w_searched) { write_GF2_coset_128_512(rec, 4); }
        weights[w]++;

        w = popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
        if (w == w_searched) { write_GF2_coset_128_512(rec, 6); }
        weights[w]++;

        if (rec < K - 2) { linear_combinationsGF2_128_512_equal_count(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_128_512_less_count(int rec, int h) {
    for (int i = h; i < K - 1; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
        if (w < w_searched) { write_GF2_coset_128_512(rec, 0); }
        weights[w]++;

        w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
        if (w < w_searched) { write_GF2_coset_128_512(rec, 2); }
        weights[w]++;

        w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]);
        if (w < w_searched) { write_GF2_coset_128_512(rec, 4); }
        weights[w]++;

        w = popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
        if (w < w_searched) { write_GF2_coset_128_512(rec, 6); }
        weights[w]++;

        if (rec < K - 2) { linear_combinationsGF2_128_512_less_count(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_64_512_equal(int rec, int h) {
    if (equal_flag) {
        for (int i = h; i < K - 1; i++) {
            reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]); if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][1]); if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][2]); if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][3]); if (w == w_searched) { equal_flag = false; break; }
            weights[w]++;

            if (rec < K - 2) { linear_combinationsGF2_64_512_equal(rec + 1, i + 1); }
        }
    }
}

void linear_combinationsGF2_64_512_equal_count(int rec, int h) {
    for (int i = h; i < K - 1; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]); if (w == w_searched) { write_GF2_coset_64_512(rec, 0); }
        weights[w]++;

        w = popcount(helper_GF2[rec][1]); if (w == w_searched) { write_GF2_coset_64_512(rec, 1); }
        weights[w]++;

        w = popcount(helper_GF2[rec][2]); if (w == w_searched) { write_GF2_coset_64_512(rec, 2); }
        weights[w]++;

        w = popcount(helper_GF2[rec][3]); if (w == w_searched) { write_GF2_coset_64_512(rec, 3); }
        weights[w]++;

        if (rec < K - 2) { linear_combinationsGF2_64_512_equal_count(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_64_512_less_count(int rec, int h) {
    for (int i = h; i < K - 1; i++) {
        reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
        unsigned long long int w = 0;
        w = popcount(helper_GF2[rec][0]); if (w < w_searched) { write_GF2_coset_64_512(rec, 0); }
        weights[w]++;

        w = popcount(helper_GF2[rec][1]); if (w < w_searched) { write_GF2_coset_64_512(rec, 1); }
        weights[w]++;

        w = popcount(helper_GF2[rec][2]); if (w < w_searched) { write_GF2_coset_64_512(rec, 2); }
        weights[w]++;

        w = popcount(helper_GF2[rec][3]); if (w < w_searched) { write_GF2_coset_64_512(rec, 3); }
        weights[w]++;

        if (rec < K - 2) { linear_combinationsGF2_64_512_less_count(rec + 1, i + 1); }
    }
}

void linear_combinationsGF2_64_512_less(int rec, int h) {
    if (less_than_flag) {
        for (int i = h; i < K - 1; i++) {
            reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]); if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][1]); if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][2]); if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][3]); if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            if (rec < K - 2) { linear_combinationsGF2_64_512_less(rec + 1, i + 1); }
        }
    }
}



void linear_combinationsGF2_128_512_less(int rec, int h) {
    if (less_than_flag) {
        for (int i = h; i < K - 1; i++) {
            reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]);
            if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
            if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]);
            if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
            if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            if (rec < K - 2) { linear_combinationsGF2_128_512_less(rec + 1, i + 1); }
        }
    }
}

void linear_combinationsGF2_256_512_less(int rec, int h) {
    if (less_than_flag) {
        for (int i = h; i < K; i++) {
            reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_matrix_GF2[i][0], reg512_helper_GF2[rec - 1][0]);
            unsigned long long int w = 0;
            w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]) + popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
            if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            w = popcount(helper_GF2[rec][4]) + popcount(helper_GF2[rec][5]) + popcount(helper_GF2[rec][6]) + popcount(helper_GF2[rec][7]);
            if (w < w_searched) { less_than_flag = false; break; }
            weights[w]++;

            if (rec < K - 1) { linear_combinationsGF2_256_512_less(rec + 1, i + 1); }
        }
    }

}

void linear_combinations_512_less(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_512(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_512(rec);
            if (w < w_searched) {
                less_than_flag = false; break;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_less(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_512_less_f(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_512(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_512_f(rec);
            if (w < w_searched) {
                less_than_flag = false; break;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_less_f(rec + 1, j + 1);
            }
        }
    }
}

// -----------------------Functions GF2-------------------------//


//--------------------------characteristic 3-------------------------//

//-----------GF3--------------------//

void setMatrixGF3_512(dynamic_mat_short& bits) {
    int c = (((N - 1) / 64) + 1);
    int bit1 = 0;

    // e.g. if we need 3 64-bit computer words for the given n, we will need 2 128-bit registers
    // writing scheme:
    // |    first bit of the representation    |    second bit of the representation   |
    // | 64 bits | 64 bits | 64 bits | ------- | 64 bits | 64 bits | 64 bits | ------- |
    // |      128 bits     |      128 bits     |      128 bits     |      128 bits     |

    if (c < 3 || c == 4) {
        bit1 = c;
    }
    else if (c == 3) {
        bit1 = 4;
    }
    else {
        bit1 = 8 * register_elements;
    }

    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3 / 8; el++) {
        reg512_matrix_CH3[0][el] = _mm512_setzero_si512();
        reg512_helper_CH3[0][el] = _mm512_set_epi64(zero, zero, zero, zero, zero, zero, zero, zero);//all 1 vector
    }

    for (int row = 1; row <= K; row++) {
        for (int i = 0; i < N_CH3 / 8; i++) {
            reg512_matrix_CH3[row][i] = _mm512_setzero_si512();
            reg512_helper_CH3[row][i] = _mm512_setzero_si512();
        }
        for (int el = 0; el < c; el++) {
            matrix_CH3[row][el] = bits.a[row - 1][el];
            matrix_CH3[row][el + bit1] = bits.a[row - 1][el + c];
        }

    }

}

static inline void add_GF3_64_512(int j, int rec, int res) { //?
    __m512i xor_1 = _mm512_setzero_si512();
    __m512i xor_2 = _mm512_setzero_si512();
    __m512i xor_rev = _mm512_setzero_si512();

    __m512i xor_rev2 = _mm512_setzero_si512();

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][0], reg512_helper_CH3[rec][0]);
    xor_rev2 = _mm512_permutex_epi64(reg512_matrix_CH3[j][0], 177);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutex_epi64(xor_1, 177);
    reg512_helper_CH3[res][0] = _mm512_or_si512(xor_2, xor_rev);
}

static inline void add_GF3_128_512(int j, int rec, int res) {
    __m512i xor_1 = _mm512_setzero_si512();
    __m512i xor_2 = _mm512_setzero_si512();
    __m512i xor_rev = _mm512_setzero_si512();

    __m512i xor_rev2 = _mm512_setzero_si512();

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][0], reg512_helper_CH3[rec][0]);
    xor_rev2 = _mm512_permutex_epi64(reg512_matrix_CH3[j][0], 78);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutex_epi64(xor_1, 78);
    reg512_helper_CH3[res][0] = _mm512_or_si512(xor_2, xor_rev);
}

static inline void add_GF3_256_512(int j, int rec, int res) {
    __m512i xor_1 = _mm512_setzero_si512();
    __m512i xor_2 = _mm512_setzero_si512();
    __m512i xor_rev = _mm512_setzero_si512();
    __m512i perm = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i xor_rev2 = _mm512_setzero_si512();

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][0], reg512_helper_CH3[rec][0]);
    xor_rev2 = _mm512_permutexvar_epi64(perm, reg512_matrix_CH3[j][0]);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutexvar_epi64(perm, xor_1);
    reg512_helper_CH3[res][0] = _mm512_or_si512(xor_2, xor_rev);
}

static inline unsigned long long int weight_GF3_64_512(int res, int pos, int bit1) {
    unsigned long long int w = 0;
    unsigned long long  w_and = helper_CH3[res][pos] ^ helper_CH3[res][pos + bit1];
    w = popcount(w_and);
    return w;
}

static inline unsigned long long int weight_GF3_64(int res) {
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    unsigned long long int w = 0;
    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 177);
    __m512i xor_r = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    w = weight_P64(xor_r);
    //w_r = _mm512_popcnt_epi64(xor_r);
    //w = w64[0];
    return w;
}

static inline unsigned long long int weight_GF3_64_f(int res) {
    static union {
        __m512i xor_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    unsigned long long int w = 0;
    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 177);
    xor_r = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    w = popcount(w64[0]);

    return w;
}


static inline unsigned long long int weight_GF3_256(int res) {
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm_m = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    unsigned long long int w = 0;
    __m512i perm = _mm512_permutexvar_epi64(perm_m, reg512_helper_CH3[res][0]);
    __m512i xor_r = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    w = weight_P256(xor_r);
    //w_r = _mm512_popcnt_epi64(xor_r);
    //w = _mm512_reduce_add_epi64(w_r) >> 1;

    return w;
}

static inline unsigned long long int weight_GF3_256_f(int res) {
    static union {
        __m512i xor_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm_m = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    unsigned long long int w = 0;
    __m512i perm = _mm512_permutexvar_epi64(perm_m, reg512_helper_CH3[res][0]);
    xor_r = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    w = popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]);

    return w;
}

static inline unsigned long long int weight_GF3_128(int res) {
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    unsigned long long int w = 0;
    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 78);
    __m512i xor_r = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    w = weight_P128(xor_r);
   // w_r = _mm512_popcnt_epi64(xor_r);
   // w = w64[0] + w64[1];

    return w;
}

static inline unsigned long long int weight_GF3_128_f(int res) {
    static union {
        __m512i xor_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    unsigned long long int w = 0;
    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 78);
    xor_r = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);

    w = popcount(w64[0]) + popcount(w64[1]);

    return w;
}

static inline unsigned long long int weight_GF3_512(int res) {
    __m512i xor_r, w_r;
    unsigned long long int w = 0;
    for (int i = 0; i < register_elements; i++) {
        xor_r = _mm512_xor_si512(reg512_helper_CH3[res][i], reg512_helper_CH3[res][i + register_elements]);
        //w_r = _mm512_popcnt_epi64(xor_r);
        w = w + weight_red_add(xor_r);//_mm512_reduce_add_epi64(w_r);
    }
    return w;
}
static inline unsigned long long int weight_GF3_512_f(int res) {
    static union {
        __m512i xor_r = _mm512_setzero_si512();
        unsigned long long int w64[8];
    };
    unsigned long long int w = 0;
    for (int i = 0; i < register_elements; i++) {
        xor_r = _mm512_xor_si512(reg512_helper_CH3[res][i], reg512_helper_CH3[res][i + register_elements]);
        w = w + popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]) +
            popcount(w64[4]) + popcount(w64[5]) + popcount(w64[6]) + popcount(w64[7]);
    }
    return w;
}

static inline void add_GF3_512(int j, int rec, int res) {
    __m512i xor_1[2];
    __m512i xor_2[2];


    for (int el = 0; el < register_elements; el++) {
        xor_1[0] = _mm512_xor_si512(reg512_matrix_CH3[j][el], reg512_helper_CH3[rec][el]);
        xor_1[1] = _mm512_xor_si512(reg512_matrix_CH3[j][el + register_elements], reg512_helper_CH3[rec][el + register_elements]);

        xor_2[0] = _mm512_xor_si512(xor_1[0], reg512_matrix_CH3[j][el + register_elements]);
        xor_2[1] = _mm512_xor_si512(xor_1[1], reg512_matrix_CH3[j][el]);

        reg512_helper_CH3[res][el] = _mm512_or_si512(xor_2[0], xor_1[1]);
        reg512_helper_CH3[res][el + register_elements] = _mm512_or_si512(xor_2[1], xor_1[0]);
    }

}

void linear_comb_recGF3_64_512(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_64(rec);
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_64_512(rec + 1, j + 1);
            }
        }
    }
}

void linear_comb_recGF3_64_512_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_64_f(rec);
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_64_512_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_comb_recGF3_128_512(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_128_512(j, rec - 2 + q1, rec);
            //int w = weight_GF3_64_512(rec, 0,2) + weight_GF3_64_512(rec, 1, 2);
            unsigned long long int w = weight_GF3_128(rec);

            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_128_512(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_128_512_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_128_512(j, rec - 2 + q1, rec);
            //int w = weight_GF3_64_512(rec, 0,2) + weight_GF3_64_512(rec, 1, 2);
            unsigned long long int w = weight_GF3_128_f(rec);

            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_128_512_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_192_512(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            //int w = weight_GF3_64_512(rec, 0, 4) + weight_GF3_64_512(rec, 1, 4) + weight_GF3_64_512(rec, 2, 4);
            unsigned long long int w = weight_GF3_256(rec);

            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_192_512(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_192_512_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            //int w = weight_GF3_64_512(rec, 0, 4) + weight_GF3_64_512(rec, 1, 4) + weight_GF3_64_512(rec, 2, 4);
            unsigned long long int w = weight_GF3_256_f(rec);

            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_192_512_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_256_512(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256(rec);
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_256_512(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_256_512_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256_f(rec);
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_256_512_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_512(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_512(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_512(rec);
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_512(rec + 1, j + 1);
            }
        }

    }
}
void linear_comb_recGF3_512_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_512(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_512_f(rec);
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_512_f(rec + 1, j + 1);
            }
        }

    }
}

//---------equal functions

void linear_comb_recGF3_64_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_64(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_64_512_equal(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_64_512_equal_count(int rec, int h) {

    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_64(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_64_512_equal_count(rec + 1, j + 1);
            }
        }

    }

}

void linear_comb_recGF3_64_512_less_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_64(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_64_512_less_count(rec + 1, j + 1);
            }
        }

    }

}

void linear_comb_recGF3_64_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_64_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_64_512_equal_f(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_64_512_equal_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_64_f(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_64_512_equal_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_64_512_less_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_64_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_64_f(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_64_512_less_count_f(rec + 1, j + 1);
            }
        }

    }
}


void linear_comb_recGF3_128_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_128_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_128(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_128_512_equal(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_128_512_equal_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_128_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_128(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_128_512_equal_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_128_512_less_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_128_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_128(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_128_512_less_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_128_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_128_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_128_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_128_512_equal_f(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_128_512_equal_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_128_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_128_f(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_128_512_equal_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_128_512_less_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_128_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_128_f(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_128_512_less_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_192_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_256(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_192_512_equal(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_192_512_equal_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_192_512_equal_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_192_512_less_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_192_512_less_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_192_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_256_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_192_512_equal_f(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_192_512_equal_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256_f(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_192_512_equal_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_192_512_less_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256_f(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_192_512_less_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_256_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_256(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_256_512_equal(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_256_512_equal_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_256_512_equal_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_256_512_less_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_256_512_less_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_256_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_256_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_256_512_equal_f(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_256_512_equal_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256_f(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_256_512_equal_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_256_512_less_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_256_512(j, rec - 2 + q1, rec);
            unsigned long long int w = weight_GF3_256_f(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_comb_recGF3_256_512_less_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_512(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_512(rec);
                if (w == w_searched) {
                    equal_flag = false; break;
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_512_equal(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_512_equal_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_512(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_512(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_512_equal_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_512_less_count(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_512(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_512(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_512_less_count(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_512(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_512_equal_f(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_512_equal_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_512(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_512_f(rec);
            if (w == w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_512_equal_count_f(rec + 1, j + 1);
            }
        }

    }
}

void linear_comb_recGF3_512_less_count_f(int rec, int h) {
    int qf = 2;
    if (h == 1) { qf = 1; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            add_GF3_512(j, rec - 2 + q1, rec);
            unsigned long long int  w = weight_GF3_512_f(rec);
            if (w < w_searched) {
                write_GF3_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_comb_recGF3_512_less_count_f(rec + 1, j + 1);
            }
        }

    }
}

//----less than functions

void linear_comb_recGF3_64_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_64(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_64_512_less(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_64_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_64_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_64_512_less_f(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_128_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_128_512(j, rec - 2 + q1, rec);
                //int w = weight_GF3_64_512(rec, 0,2) + weight_GF3_64_512(rec, 1, 2);
                unsigned long long int w = weight_GF3_128(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_128_512_less(rec + 1, j + 1);
                }
            }

        }
    }
}
void linear_comb_recGF3_128_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_128_512(j, rec - 2 + q1, rec);
                //int w = weight_GF3_64_512(rec, 0,2) + weight_GF3_64_512(rec, 1, 2);
                unsigned long long int w = weight_GF3_128_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_128_512_less_f(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_192_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                //int w = weight_GF3_64_512(rec, 0, 4) + weight_GF3_64_512(rec, 1, 4) + weight_GF3_64_512(rec, 2, 4);
                unsigned long long int w = weight_GF3_256(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_192_512_less(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_192_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                //int w = weight_GF3_64_512(rec, 0, 4) + weight_GF3_64_512(rec, 1, 4) + weight_GF3_64_512(rec, 2, 4);
                unsigned long long int w = weight_GF3_256_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_192_512_less_f(rec + 1, j + 1);
                }
            }

        }
    }
}
void linear_comb_recGF3_256_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_256(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_256_512_less(rec + 1, j + 1);
                }
            }

        }
    }
}

void linear_comb_recGF3_256_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_256_512(j, rec - 2 + q1, rec);
                unsigned long long int w = weight_GF3_256_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_comb_recGF3_256_512_less_f(rec + 1, j + 1);
                }
            }

        }
    }
}


void linear_comb_recGF3_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_512(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_512_less(rec + 1, j + 1);
                }
            }

        }
    }
}
void linear_comb_recGF3_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_512(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_512_less_f(rec + 1, j + 1);
                }
            }

        }
    }
}



//------------GF3-------------------//

//-----------GF9--------------------//

void setMatrixGF9_512(dynamic_mat_short& bits) {
    int c = (((N - 1) / 64) + 1);

    int bit1 = 0;
    if (c < 3 || c == 4) {
        bit1 = c;
    }
    else if (c == 3) {
        bit1 = 4;
    }
    else {
        bit1 = 8 * register_elements;
    }



    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3 / 8; el++) {
        reg512_matrix_CH3[0][el] = _mm512_setzero_si512();
        reg512_helper_CH3[0][el] = _mm512_set1_epi64(zero);//all 1 vector
    }

    for (int row = 1; row <= 2 * K; row++) {
        for (int i = 0; i < N_CH3 / 8; i++) {
            reg512_matrix_CH3[row][i] = _mm512_setzero_si512();
            reg512_helper_CH3[row][i] = _mm512_setzero_si512();
        }
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


static inline void addGF9_64_512(int j, int rec, int res) {
    __m512i xor_1 = _mm512_setzero_si512();
    __m512i xor_2 = _mm512_setzero_si512();
    __m512i xor_rev = _mm512_setzero_si512();
    __m512i xor_rev2 = _mm512_setzero_si512();

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][0], reg512_helper_CH3[rec][0]);
    xor_rev2 = _mm512_permutex_epi64(reg512_matrix_CH3[j][0], 177);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutex_epi64(xor_1, 177);
    reg512_helper_CH3[res][0] = _mm512_or_si512(xor_2, xor_rev);

}

static inline void addGF9_128_512(int j, int rec, int res) {
    __m512i xor_1 = _mm512_setzero_si512();
    __m512i xor_2 = _mm512_setzero_si512();
    __m512i xor_rev = _mm512_setzero_si512();
    __m512i xor_rev2 = _mm512_setzero_si512();

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][0], reg512_helper_CH3[rec][0]);
    xor_rev2 = _mm512_permutex_epi64(reg512_matrix_CH3[j][0], 78);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutex_epi64(xor_1, 78);
    reg512_helper_CH3[res][0] = _mm512_or_si512(xor_2, xor_rev);

}

static inline void addGF9_256_512(int j, int rec, int res) {
    __m512i xor_1 = _mm512_setzero_si512();
    __m512i xor_2 = _mm512_setzero_si512();
    __m512i xor_rev = _mm512_setzero_si512();
    __m512i xor_rev2 = _mm512_setzero_si512();
    __m512i perm_m = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][0], reg512_helper_CH3[rec][0]);
    xor_rev2 = _mm512_permutexvar_epi64(perm_m, reg512_matrix_CH3[j][0]);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutexvar_epi64(perm_m, xor_1);
    reg512_helper_CH3[res][0] = _mm512_or_si512(xor_2, xor_rev);

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][1], reg512_helper_CH3[rec][1]);
    xor_rev2 = _mm512_permutexvar_epi64(perm_m, reg512_matrix_CH3[j][1]);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutexvar_epi64(perm_m, xor_1);
    reg512_helper_CH3[res][1] = _mm512_or_si512(xor_2, xor_rev);
}


static inline void addGF9_512(int j, int rec, int res) {
    __m512i xor_1[2];// = _mm_setzero_si128();
    __m512i xor_2[2];// = _mm_setzero_si128();

    xor_1[0] = _mm512_setzero_si512();
    xor_1[1] = _mm512_setzero_si512();
    xor_2[0] = _mm512_setzero_si512();
    xor_2[1] = _mm512_setzero_si512();
    for (int i = 0; i < register_elements; i++) {
        xor_1[0] = _mm512_xor_si512(reg512_matrix_CH3[j][i], reg512_helper_CH3[rec][i]);
        xor_1[1] = _mm512_xor_si512(reg512_matrix_CH3[j][i + register_elements], reg512_helper_CH3[rec][i + register_elements]);

        xor_2[0] = _mm512_xor_si512(xor_1[0], reg512_matrix_CH3[j][i + register_elements]);
        xor_2[1] = _mm512_xor_si512(xor_1[1], reg512_matrix_CH3[j][i]);

        reg512_helper_CH3[res][i] = _mm512_or_si512(xor_2[0], xor_1[1]);
        reg512_helper_CH3[res][i + register_elements] = _mm512_or_si512(xor_2[1], xor_1[0]);

        xor_1[0] = _mm512_xor_si512(reg512_matrix_CH3[j][i + 2 * register_elements], reg512_helper_CH3[rec][i + 2 * register_elements]);
        xor_1[1] = _mm512_xor_si512(reg512_matrix_CH3[j][i + 3 * register_elements], reg512_helper_CH3[rec][i + 3 * register_elements]);

        xor_2[0] = _mm512_xor_si512(xor_1[0], reg512_matrix_CH3[j][i + 3 * register_elements]);
        xor_2[1] = _mm512_xor_si512(xor_1[1], reg512_matrix_CH3[j][i + 2 * register_elements]);

        reg512_helper_CH3[res][i + 2 * register_elements] = _mm512_or_si512(xor_2[0], xor_1[1]);
        reg512_helper_CH3[res][i + 3 * register_elements] = _mm512_or_si512(xor_2[1], xor_1[0]);
    }
}


unsigned long long int weightGF9_64_512(int res) {
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 177);
    __m512i xor1 = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    perm = _mm512_permutex_epi64(xor1, 78);
    __m512i or1 = _mm512_or_si512(xor1, perm);
    //w_r = _mm512_popcnt_epi64(or1);
    unsigned long long int w = weight_P64(or1); //w64[0];
    return w;
}

unsigned long long int weightGF9_64_512_f(int res) {
    static union {
        __m512i or1;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 177);
    __m512i xor1 = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    perm = _mm512_permutex_epi64(xor1, 78);
    or1 = _mm512_or_si512(xor1, perm);

    unsigned long long int w = popcount(w64[0]);
    return w;
}

unsigned long long int weightGF9_128_512(int res) {
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm_m = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 78);
    __m512i xor1 = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    perm = _mm512_permutexvar_epi64(perm_m, xor1);
    __m512i or1 = _mm512_or_si512(xor1, perm);
   // w_r = _mm512_popcnt_epi64(or1);
    unsigned long long int w = weight_P128(or1); //w64[0] + w64[1];
    return w;
}

unsigned long long int weightGF9_128_512_f(int res) {
    static union {
        __m512i or1;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm_m = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    __m512i perm = _mm512_permutex_epi64(reg512_helper_CH3[res][0], 78);
    __m512i xor1 = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);
    perm = _mm512_permutexvar_epi64(perm_m, xor1);
    or1 = _mm512_or_si512(xor1, perm);

    unsigned long long int w = popcount(w64[0]) + popcount(w64[1]);
    return w;
}

unsigned long long int weightGF9_256_512(int res) {
    static union {
        __m512i w_r;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm_m = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    __m512i perm = _mm512_permutexvar_epi64(perm_m, reg512_helper_CH3[res][0]);
    __m512i xor1 = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);

    perm = _mm512_permutexvar_epi64(perm_m, reg512_helper_CH3[res][1]);
    __m512i xor2 = _mm512_xor_si512(perm, reg512_helper_CH3[res][1]);

    __m512i or1 = _mm512_or_si512(xor1, xor2);
    //w_r = _mm512_popcnt_epi64(or1);
    unsigned long long int w = weight_P256(or1); //_mm512_reduce_add_epi64(w_r) >> 1;// w64[0] + w64[1];
    return w;
}

unsigned long long int weightGF9_256_512_f(int res) {
    static union {
        __m512i or1;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm_m = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

    __m512i perm = _mm512_permutexvar_epi64(perm_m, reg512_helper_CH3[res][0]);
    __m512i xor1 = _mm512_xor_si512(perm, reg512_helper_CH3[res][0]);

    perm = _mm512_permutexvar_epi64(perm_m, reg512_helper_CH3[res][1]);
    __m512i xor2 = _mm512_xor_si512(perm, reg512_helper_CH3[res][1]);

    or1 = _mm512_or_si512(xor1, xor2);

    unsigned long long int w = popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]);;
    return w;
}

static inline unsigned long long int weight_GF9_512(int res) {
    __m512i element1 = _mm512_setzero_si512(), element2 = _mm512_setzero_si512(), w_r = _mm512_setzero_si512();
    static union {
        __m512i temp = _mm512_setzero_si512();
        unsigned long long  temp64[8];
    };
    unsigned long long int count = 0;

    for (int i = 0; i < register_elements; i++) {
        element1 = _mm512_xor_si512(reg512_helper_CH3[res][i], reg512_helper_CH3[res][i + register_elements]);
        element2 = _mm512_xor_si512(reg512_helper_CH3[res][i + 2 * register_elements], reg512_helper_CH3[res][i + 3 * register_elements]);
        temp = _mm512_or_si512(element1, element2);
       // w_r = _mm512_popcnt_epi64(temp);
        count = count + weight_red_add(temp); //_mm512_reduce_add_epi64(w_r);
    }
    return count;
}

static inline unsigned long long int weight_GF9_512_f(int res) {
    __m512i element1 = _mm512_setzero_si512(), element2 = _mm512_setzero_si512();
    static union {
        __m512i temp = _mm512_setzero_si512();
        unsigned long long  temp64[8];
    };
    unsigned long long int count = 0;

    for (int i = 0; i < register_elements; i++) {
        element1 = _mm512_xor_si512(reg512_helper_CH3[res][i], reg512_helper_CH3[res][i + register_elements]);
        element2 = _mm512_xor_si512(reg512_helper_CH3[res][i + 2 * register_elements], reg512_helper_CH3[res][i + 3 * register_elements]);
        temp = _mm512_or_si512(element1, element2);
        count = count + popcount(temp64[0]) + popcount(temp64[1]) + popcount(temp64[2]) + popcount(temp64[3]) +
            popcount(temp64[4]) + popcount(temp64[5]) + popcount(temp64[6]) + popcount(temp64[7]);
    }
    return count;
}


void linear_combinations_GF9_64_512(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_512(t * K + j, rec, rec);
            }
            unsigned long long weight = weightGF9_64_512(rec);
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_64_512(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_64_512_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_512(t * K + j, rec, rec);
            }
            unsigned long long weight = weightGF9_64_512_f(rec);
            weights[weight]++;
            if (rec < K) {
                linear_combinations_GF9_64_512_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_512(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_128_512(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_128_512(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_512_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_128_512_f(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_128_512_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_256_512(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_256_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_256_512(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_256_512(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_256_512_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_256_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_256_512_f(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_256_512_f(rec + 1, j + 1);
            }
        }
    }
}


void linear_combinations_GF9_512(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_512(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_512(rec + 1, j + 1);
            }
        }
    }
}
void linear_combinations_GF9_512_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_512_f(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_512_f(rec + 1, j + 1);
            }
        }
    }
}

//----------------- less than

void linear_combinations_GF9_64_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_64_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_64_512_less(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_64_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_64_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_64_512_less_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_128_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_128_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_128_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_128_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_128_512_less(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_128_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_128_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_128_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_128_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_128_512_less_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_256_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_256_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_256_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_256_512_less(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_256_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_256_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_256_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_256_512_less_f(rec + 1, j + 1);
                }
            }
        }
    }
}


void linear_combinations_GF9_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_512_less(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_512_less_f(rec + 1, j + 1);
                }
            }
        }
    }
}
//--------------------- equal


void linear_combinations_GF9_64_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_64_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_64_512_equal(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_64_512_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_64_512(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_64_512_equal_count(rec + 1, j + 1);
            }
        }
    }
}


void linear_combinations_GF9_64_512_less_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_64_512(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_64_512_less_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_64_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_64_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_64_512_equal_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_64_512_equal_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_64_512_f(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_64_512_equal_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_64_512_less_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_64_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_64_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_64_512_f(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_64_512_less_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_128_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_128_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_128_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_128_512_equal(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_128_512_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_128_512(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_128_512_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_512_less_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_128_512(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_128_512_less_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_128_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_128_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_128_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_128_512_equal_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_128_512_equal_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_128_512_f(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_128_512_equal_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_128_512_less_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_128_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_128_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_128_512_f(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_128_512_less_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_256_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_256_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_256_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_256_512_equal(rec + 1, j + 1);
                }
            }
        }
    }
}


void linear_combinations_GF9_256_512_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_256_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_256_512(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_256_512_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_256_512_less_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_256_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_256_512(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_256_512_less_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_256_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_256_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weightGF9_256_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_256_512_equal_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_256_512_equal_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_256_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_256_512_f(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_256_512_equal_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_256_512_less_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_256_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weightGF9_256_512_f(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_256_512_less_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_512_equal(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_512_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_512(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_512_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_512_less_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_512(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_512_less_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_512(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_512_equal_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_512_equal_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_512_f(rec);
            if (w == w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_512_equal_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF9_512_less_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 8; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                addGF9_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                addGF9_512(t * K + j, rec, rec);
            }
            unsigned long long int w = weight_GF9_512_f(rec);
            if (w < w_searched) {
                write_GF9_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF9_512_less_count_f(rec + 1, j + 1);
            }
        }
    }
}

//----------GF9---------------------//

//-----------GF27-------------------//


void setMatrixGF27_512(dynamic_mat_short& bits) {

    int c = (((N - 1) / 64) + 1);
    register_elements = ((N - 1) / 512) + 1;
    int bit1 = 0;
    if (c <= 4) {
        bit1 = 4;
    }
    else {
        bit1 = 8 * register_elements;
    }

    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3 / 8; el++) {
        reg512_matrix_CH3[0][el] = _mm512_setzero_si512();
        reg512_helper_CH3[0][el] = _mm512_set1_epi64(zero);//all 1 vector
    }

    for (int row = 1; row <= 3 * K; row++) {
        for (int i = 0; i < N_CH3 / 8; i++) {
            reg512_matrix_CH3[row][i] = _mm512_setzero_si512();
            reg512_helper_CH3[row][i] = _mm512_setzero_si512();
        }



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
        //cout << endl;
    }
}

static inline void add_GF27_256_512(int j, int rec, int res) { //?
    __m512i xor_1 = _mm512_setzero_si512();
    __m512i xor_2 = _mm512_setzero_si512();
    __m512i xor_rev = _mm512_setzero_si512();
    __m512i perm = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i xor_rev2 = _mm512_setzero_si512();

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][0], reg512_helper_CH3[rec][0]);
    xor_rev2 = _mm512_permutexvar_epi64(perm, reg512_matrix_CH3[j][0]);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutexvar_epi64(perm, xor_1);
    reg512_helper_CH3[res][0] = _mm512_or_si512(xor_2, xor_rev);

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][1], reg512_helper_CH3[rec][1]);
    xor_rev2 = _mm512_permutexvar_epi64(perm, reg512_matrix_CH3[j][1]);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutexvar_epi64(perm, xor_1);
    reg512_helper_CH3[res][1] = _mm512_or_si512(xor_2, xor_rev);

    xor_1 = _mm512_xor_si512(reg512_matrix_CH3[j][2], reg512_helper_CH3[rec][2]);
    xor_rev2 = _mm512_permutexvar_epi64(perm, reg512_matrix_CH3[j][2]);
    xor_2 = _mm512_xor_si512(xor_1, xor_rev2);
    xor_rev = _mm512_permutexvar_epi64(perm, xor_1);
    reg512_helper_CH3[res][2] = _mm512_or_si512(xor_2, xor_rev);
}


static inline void add_GF27_512(int j, int rec, int res) {
    __m512i xor_1[2];// = _mm_setzero_si128();
    __m512i xor_2[2];// = _mm_setzero_si128();


    xor_1[0] = _mm512_setzero_si512();
    xor_1[1] = _mm512_setzero_si512();
    xor_2[0] = _mm512_setzero_si512();
    xor_2[1] = _mm512_setzero_si512();

    for (int i = 0; i < register_elements; i++) {
        xor_1[0] = _mm512_xor_si512(reg512_matrix_CH3[j][i], reg512_helper_CH3[rec][i]);
        xor_1[1] = _mm512_xor_si512(reg512_matrix_CH3[j][i + register_elements], reg512_helper_CH3[rec][i + register_elements]);
        xor_2[0] = _mm512_xor_si512(xor_1[0], reg512_matrix_CH3[j][i + register_elements]);
        xor_2[1] = _mm512_xor_si512(xor_1[1], reg512_matrix_CH3[j][i]);

        reg512_helper_CH3[res][i] = _mm512_or_si512(xor_2[0], xor_1[1]);
        reg512_helper_CH3[res][i + register_elements] = _mm512_or_si512(xor_2[1], xor_1[0]);

        xor_1[0] = _mm512_xor_si512(reg512_matrix_CH3[j][i + 2 * register_elements], reg512_helper_CH3[rec][i + 2 * register_elements]);
        xor_1[1] = _mm512_xor_si512(reg512_matrix_CH3[j][i + 3 * register_elements], reg512_helper_CH3[rec][i + 3 * register_elements]);

        xor_2[0] = _mm512_xor_si512(xor_1[0], reg512_matrix_CH3[j][i + 3 * register_elements]);
        xor_2[1] = _mm512_xor_si512(xor_1[1], reg512_matrix_CH3[j][i + 2 * register_elements]);

        reg512_helper_CH3[res][i + 2 * register_elements] = _mm512_or_si512(xor_2[0], xor_1[1]);
        reg512_helper_CH3[res][i + 3 * register_elements] = _mm512_or_si512(xor_2[1], xor_1[0]);

        xor_1[0] = _mm512_xor_si512(reg512_matrix_CH3[j][i + 4 * register_elements], reg512_helper_CH3[rec][i + 4 * register_elements]);
        xor_1[1] = _mm512_xor_si512(reg512_matrix_CH3[j][i + 5 * register_elements], reg512_helper_CH3[rec][i + 5 * register_elements]);

        xor_2[0] = _mm512_xor_si512(xor_1[0], reg512_matrix_CH3[j][i + 5 * register_elements]);
        xor_2[1] = _mm512_xor_si512(xor_1[1], reg512_matrix_CH3[j][i + 4 * register_elements]);
        reg512_helper_CH3[res][i + 4 * register_elements] = _mm512_or_si512(xor_2[0], xor_1[1]);
        reg512_helper_CH3[res][i + 5 * register_elements] = _mm512_or_si512(xor_2[1], xor_1[0]);
    }
}



static inline unsigned long long int weight_GF27_256_512(int pos) {
    __m512i perm = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i perm_res = _mm512_permutexvar_epi64(perm, reg512_helper_CH3[pos][0]);
    __m512i xor1 = _mm512_xor_si512(perm_res, reg512_helper_CH3[pos][0]);

    perm_res = _mm512_permutexvar_epi64(perm, reg512_helper_CH3[pos][1]);
    __m512i xor2 = _mm512_xor_si512(perm_res, reg512_helper_CH3[pos][1]);

    perm_res = _mm512_permutexvar_epi64(perm, reg512_helper_CH3[pos][2]);
    __m512i xor3 = _mm512_xor_si512(perm_res, reg512_helper_CH3[pos][2]);

    __m512i or1 = _mm512_or_si512(xor1, xor2);
    or1 = _mm512_or_si512(or1, xor3);


   // __m512i w_r = _mm512_popcnt_epi64(or1);
    unsigned long long int w = weight_P256(or1); //_mm512_reduce_add_epi64(w_r) >> 1;
    return w;
}

static inline unsigned long long int weight_GF27_256_512_f(int pos) {
    static union {
        __m512i or1;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i perm_res = _mm512_permutexvar_epi64(perm, reg512_helper_CH3[pos][0]);
    __m512i xor1 = _mm512_xor_si512(perm_res, reg512_helper_CH3[pos][0]);

    perm_res = _mm512_permutexvar_epi64(perm, reg512_helper_CH3[pos][1]);
    __m512i xor2 = _mm512_xor_si512(perm_res, reg512_helper_CH3[pos][1]);

    perm_res = _mm512_permutexvar_epi64(perm, reg512_helper_CH3[pos][2]);
    __m512i xor3 = _mm512_xor_si512(perm_res, reg512_helper_CH3[pos][2]);

    or1 = _mm512_or_si512(xor1, xor2);
    or1 = _mm512_or_si512(or1, xor3);
    unsigned long long int w = popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]);

    return w;
}

static inline unsigned long long int weight_GF27_512(int res) {
    __m512i element1 = _mm512_setzero_si512(), element2 = _mm512_setzero_si512(), element3 = _mm512_setzero_si512();
    __m512i temp = _mm512_setzero_si512(), w_r = _mm512_setzero_si512();
    unsigned long long int count = 0;

    for (int i = 0; i < register_elements; i++) {
        element1 = _mm512_xor_si512(reg512_helper_CH3[res][i], reg512_helper_CH3[res][i + register_elements]);
        element2 = _mm512_xor_si512(reg512_helper_CH3[res][i + 2 * register_elements], reg512_helper_CH3[res][i + 3 * register_elements]);
        element3 = _mm512_xor_si512(reg512_helper_CH3[res][i + 4 * register_elements], reg512_helper_CH3[res][i + 5 * register_elements]);
        temp = _mm512_or_si512(element1, element2);
        temp = _mm512_or_si512(temp, element3);
        //w_r = _mm512_popcnt_epi64(temp);
        count = count + weight_red_add(temp); //_mm512_reduce_add_epi64(w_r);
    }
    return count;
}

static inline unsigned long long int weight_GF27_512_f(int res) {
    __m512i element1 = _mm512_setzero_si512(), element2 = _mm512_setzero_si512(), element3 = _mm512_setzero_si512();
    static union {
        __m512i temp;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    unsigned long long int count = 0;

    for (int i = 0; i < register_elements; i++) {
        element1 = _mm512_xor_si512(reg512_helper_CH3[res][i], reg512_helper_CH3[res][i + register_elements]);
        element2 = _mm512_xor_si512(reg512_helper_CH3[res][i + 2 * register_elements], reg512_helper_CH3[res][i + 3 * register_elements]);
        element3 = _mm512_xor_si512(reg512_helper_CH3[res][i + 4 * register_elements], reg512_helper_CH3[res][i + 5 * register_elements]);
        temp = _mm512_or_si512(element1, element2);
        temp = _mm512_or_si512(temp, element3);

        count = count + popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]) +
            popcount(w64[4]) + popcount(w64[5]) + popcount(w64[6]) + popcount(w64[7]);
    }
    return count;
}


void linear_combinations_GF27_256_512(int rec, int h) {

    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_256_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_256_512(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_256_512(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_256_512_f(int rec, int h) {

    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_256_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_256_512_f(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_256_512_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_512(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_512(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_512(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_512_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_512_f(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_512_f(rec + 1, j + 1);
            }
        }
    }
}

//-------------less

void linear_combinations_GF27_256_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_256_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_256_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_256_512_less(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF27_256_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_256_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_256_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_256_512_less_f(rec + 1, j + 1);
                }
            }
        }
    }
}


void linear_combinations_GF27_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_512_less(rec + 1, j + 1);
                }
            }
        }
    }
}
void linear_combinations_GF27_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_512_less_f(rec + 1, j + 1);
                }
            }
        }
    }
}

//---------------equal

void linear_combinations_GF27_256_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_256_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_256_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_256_512_equal(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF27_256_512_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_256_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_256_512(rec);
            if (w == w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_256_512_equal_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_256_512_less_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_256_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_256_512(rec);
            if (w < w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_256_512_less_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_256_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_256_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_256_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_256_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_256_512_equal_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF27_256_512_equal_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_256_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_256_512_f(rec);
            if (w == w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_256_512_equal_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_256_512_less_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_256_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_256_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_256_512_f(rec);
            if (w < w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_256_512_less_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_512_equal(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF27_512_equal_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_512(rec);
            if (w == w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_512_equal_count(rec + 1, j + 1);
            }
        }
    }
}
void linear_combinations_GF27_512_less_count(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_512(rec);
            if (w < w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_512_less_count(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    add_GF27_512(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    add_GF27_512(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_512_equal_f(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF27_512_equal_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_512_f(rec);
            if (w == w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_512_equal_count_f(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_GF27_512_less_count_f(int rec, int h) {
    int qf;
    if (h == 1) { qf = 1; }
    else { qf = 26; }
    for (int j = h; j <= K; j++) {
        for (int q1 = 1; q1 <= qf; q1++) {
            if (q1 == 1) {
                add_GF27_512(j, rec - 1, rec);
            }
            else {
                int t = TransitionSequence27[q1] - 1;
                add_GF27_512(t * K + j, rec, rec);
            }

            unsigned long long int w = weight_GF27_512_f(rec);
            if (w < w_searched) {
                write_GF27_512(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_GF27_512_less_count_f(rec + 1, j + 1);
            }
        }
    }
}

//-----------GF27-------------------//

//--------------------------characteristic 3-------------------------//


//------------------------- bitwise CH2 (GF4)--------------------------//

void setMatrixGF2_CF_512(dynamic_mat_short& bits) {
    for (int i = 0; i <= N_FIX * 8; i++) {
        weights[i] = 0;
    }
    register_elements = ((N - 1) / 512) + 1;
    int c = ((N - 1) / 64) + 1;
    int bit1 = 0;
    if (N < 256) {
        bit1 = 4;
    }
    else {
        bit1 = 8 * register_elements;
    }
    for (int col = 0; col < (N_GF2 / 8); col++) {
        reg512_matrix_GF2[0][col] = _mm512_setzero_si512();
        reg512_helper_GF2[0][col] = _mm512_setzero_si512();
    }

    for (int row = 1; row <= (M * (K + 1)); row++) {
        for (int col = 0; col < (N_GF2 / 8); col++) {
            reg512_matrix_GF2[row][col] = _mm512_setzero_si512();
            reg512_helper_GF2[row][col] = _mm512_setzero_si512();
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

static inline void add_GF4_512(int rec, int i, int res) {
    unsigned long long int w = 0;
    // alignas(32)  unsigned long long int temp[4];
     //unsigned long long int* temp;
    for (int el = 0; el < 2 * register_elements; el++) {
        reg512_helper_GF2[res][el] = _mm512_xor_si512(reg512_helper_GF2[rec][el], reg512_matrix_GF2[i][el]);
    }

}

static inline unsigned long long int weight_GF4_256_512(int res) {
    __m512i perm = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i temp = _mm512_permutexvar_epi64(perm, reg512_helper_GF2[res][0]);
    __m512i or1 = _mm512_or_si512(temp, reg512_helper_GF2[res][0]);
   // temp = _mm512_popcnt_epi64(or1);
    unsigned long long int w = weight_P256(or1); //_mm512_reduce_add_epi64(temp) >> 1;
    return w;
}

static inline unsigned long long int weight_GF4_256_512_f(int res) {
    static union {
        __m512i or1;
        unsigned long long int w64[8] = { 0,0,0,0,0,0,0,0 };
    };
    __m512i perm = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i temp = _mm512_permutexvar_epi64(perm, reg512_helper_GF2[res][0]);
    or1 = _mm512_or_si512(temp, reg512_helper_GF2[res][0]);

    unsigned long long int w = popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]);
    return w;
}

static inline unsigned long long int weight_GF4_512(int res) {
    unsigned long long int w = 0;
    __m512i temp_reg = _mm512_setzero_si512();
    for (int el = 0; el < register_elements; el++) {
        temp_reg = _mm512_or_si512(reg512_helper_GF2[res][el], reg512_helper_GF2[res][register_elements + el]);
       // __m512i w_r = _mm512_popcnt_epi64(temp_reg);
        w = w + weight_red_add(temp_reg); //_mm512_reduce_add_epi64(w_r);
    }
    return w;
}

static inline unsigned long long int weight_GF4_512_f(int res) {
    unsigned long long int w = 0;
    static union {
        __m512i temp_reg = _mm512_setzero_si512();
        unsigned long long int w64[8];
    };
    for (int el = 0; el < register_elements; el++) {
        temp_reg = _mm512_or_si512(reg512_helper_GF2[res][el], reg512_helper_GF2[res][register_elements + el]);

        w = w + popcount(w64[0]) + popcount(w64[1]) + popcount(w64[2]) + popcount(w64[3]) +
            popcount(w64[4]) + popcount(w64[5]) + popcount(w64[6]) + popcount(w64[7]);
    }
    return w;
}


void linear_combinations_GF4_256_512(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);
                //add_GF2_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);
                //add_GF2_256_256(rec, t * K + i, rec);
            }

            unsigned long long int weight = weight_GF4_256_512(rec);
            weights[weight]++;

            if (rec < K) {
                linear_combinations_GF4_256_512(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_256_512_f(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);
                //add_GF2_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);
                //add_GF2_256_256(rec, t * K + i, rec);
            }

            unsigned long long int weight = weight_GF4_256_512_f(rec);
            weights[weight]++;

            if (rec < K) {
                linear_combinations_GF4_256_512_f(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_512(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                add_GF4_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF4_512(rec, t * K + i, rec);
            }

            unsigned long long int weight = weight_GF4_512(rec);
            weights[weight]++;

            if (rec < K) {
                linear_combinations_GF4_512(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_512_f(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                add_GF4_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF4_512(rec, t * K + i, rec);
            }

            unsigned long long int weight = weight_GF4_512_f(rec);
            weights[weight]++;

            if (rec < K) {
                linear_combinations_GF4_512_f(rec + 1, i + 1);
            }
        }
    }
}


//---------less than


void linear_combinations_GF4_256_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);
                    //add_GF2_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);
                    //add_GF2_256_256(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_GF4_256_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_256_512_less(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_GF4_256_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);
                    //add_GF2_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);
                    //add_GF2_256_256(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_GF4_256_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_256_512_less_f(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_GF4_512_less(int rec, int h) {
    if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    add_GF4_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_GF4_512(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_512_less(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_GF4_512_less_f(int rec, int h) {
    if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    add_GF4_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_GF4_512_f(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_512_less_f(rec + 1, i + 1);
                }
            }
        }
    }
}

//---------equal

void linear_combinations_GF4_256_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);
                }

                unsigned long long int w = weight_GF4_256_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_256_512_equal(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_GF4_256_512_equal_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);
            }

            unsigned long long int w = weight_GF4_256_512(rec);
            if (w == w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_256_512_equal_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_256_512_less_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);
            }

            unsigned long long int w = weight_GF4_256_512(rec);
            if (w < w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_256_512_less_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_256_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);

                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);

                }

                unsigned long long int w = weight_GF4_256_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_256_512_equal_f(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_GF4_256_512_equal_count_f(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);

            }
            else {
                int t = TransitionSequence64[q1] - 1;
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);

            }

            unsigned long long int w = weight_GF4_256_512_f(rec);
            if (w == w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_256_512_equal_count_f(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_256_512_less_count_f(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec - 1][0], reg512_matrix_GF2[i][0]);

            }
            else {
                int t = TransitionSequence64[q1] - 1;
                reg512_helper_GF2[rec][0] = _mm512_xor_si512(reg512_helper_GF2[rec][0], reg512_matrix_GF2[t * K + i][0]);

            }

            unsigned long long int w = weight_GF4_256_512_f(rec);
            if (w < w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_256_512_less_count_f(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_512_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    add_GF4_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_GF4_512(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_512_equal(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_GF4_512_equal_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                add_GF4_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF4_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weight_GF4_512(rec);
            if (w == w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_512_equal_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_512_less_count(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                add_GF4_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF4_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weight_GF4_512(rec);
            if (w < w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_512_less_count(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_512_equal_f(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if ((q1 == 1)) {
                    add_GF4_512(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_512(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_GF4_512_f(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_GF4_512_equal_f(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_GF4_512_equal_count_f(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                add_GF4_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF4_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weight_GF4_512_f(rec);
            if (w == w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_512_equal_count_f(rec + 1, i + 1);
            }
        }
    }
}

void linear_combinations_GF4_512_less_count_f(int rec, int h) {
    int qf = 4;
    if (h == 1) { qf = 2; }
    //else { qf = 4;}
    for (int i = h; i <= K; i++) {
        for (int q1 = 1; q1 < qf; q1++) {
            if ((q1 == 1)) {
                add_GF4_512(rec - 1, i, rec);
            }
            else {
                int t = TransitionSequence64[q1] - 1;
                add_GF4_512(rec, t * K + i, rec);
            }

            unsigned long long int w = weight_GF4_512_f(rec);
            if (w < w_searched) {
                write_CF2_512(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_GF4_512_less_count_f(rec + 1, i + 1);
            }
        }
    }
}

//------------------------- bitwise CH2 (GF4)----------------------------//



//-----------------------------calculateWeight-----------------------//
void calculateWeightGF2_512(dynamic_mat_short& bits, int n, int k) {
    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;


    register_elements = ((n - 1) / 512) + 1;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }



    if (n <= 64) {
        set_64_512(bits);
        unsigned int w = popcount(helper_GF2[0][1]);
        weights[w]++;

        w = popcount(helper_GF2[0][2]);
        weights[w]++;

        w = popcount(helper_GF2[0][3]);
        weights[w]++;
        linear_combinationsGF2_64_512(1, 1);
    }
    else if (n <= 128) {
        set_128_512(bits);
        unsigned long long int  w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        weights[w]++;

        w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]);
        weights[w]++;

        w = popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        weights[w]++;

        linear_combinationsGF2_128_512(1, 1);
    }
    else if (n <= 256) {
        set_256_512(bits);
        unsigned long long int  w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]) +
            popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        weights[w]++;

        linear_combinationsGF2_256_512(1, 1);
    }
    else {
        set_512(bits);
        if ((POPCNT == 2) || (POPCNT == -2)) {
            linear_combinations_512(1, 1);
        }
        else { linear_combinations_512_f(1, 1); }


    }


}

void calculateWeightCH2_512(dynamic_mat_short& bits, int n, int k, int m) {

    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    register_elements = ((n - 1) / 512) + 1;


    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    setMatrixGF2_CF_512(bits);
    if ((POPCNT == 2) || (POPCNT == -2)) {
        if (Q == 4) { // swich () case:
            if (n <= 256) {
                linear_combinations_GF4_256_512(1, 1);
            }
            else {
                linear_combinations_GF4_512(1, 1);
            }

        }
    }
    else {
        if (Q == 4) { // swich () case:
            if (n <= 256) {
                linear_combinations_GF4_256_512_f(1, 1);
            }
            else {
                linear_combinations_GF4_512_f(1, 1);
            }

        }
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
}


void calculateWeightBytes_512(dmat_type& bits, int n, int k, int m, int q) {

    popcnt_detect();
    K = k;
    N = n;
    M = m;
    register_elements = ((n - 1) / 512) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes_512(bits);
            linear_combinations_CH2_512(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_49_512(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_25_512(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytesCF_512(bits);
        linear_combinations_Bytes_512(1, 1);
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
}

void calculateWeightCH3_512(dynamic_mat_short& bits, int n, int k, int m) {
    popcnt_detect();

    K = k;
    N = n;
    M = m;


    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    register_elements = (((n - 1) / 512) + 1);
    if ((POPCNT == 2) || (POPCNT == -2)) {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512(1, 1);
            }
            else {
                linear_comb_recGF3_512(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512(1, 1);
            }
            else {
                linear_combinations_GF9_512(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512(1, 1);
            }
            else {
                linear_combinations_GF27_512(1, 1);
            }

        }
    }
    else {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_f(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_f(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_f(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_f(1, 1);
            }
            else {
                linear_comb_recGF3_512_f(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_f(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_f(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_f(1, 1);
            }
            else {
                linear_combinations_GF9_512_f(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_f(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_f(1, 1);
            }
            else {
                linear_combinations_GF27_512_f(1, 1);
            }

        }
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }

}



//-----------------------------calculateWeight_equal--------------------------------------//
bool calculateWeightGF2_512_equal(dynamic_mat_short& bits, int n, int k, int d) {
    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    equal_flag = true;

    register_elements = ((n - 1) / 512) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (n <= 64) {
        set_64_512(bits);
        unsigned long long int w = popcount(helper_GF2[0][1]);
        if (w == w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][2]);
        if (w == w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][3]);
        if (w == w_searched) return true;
        weights[w]++;
        linear_combinationsGF2_64_512_equal(1, 1);
    }
    else if (n <= 128) {
        set_128_512(bits);

        unsigned long long int  w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        if (w == w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]);
        if (w == w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w == w_searched) return true;
        weights[w]++;

        linear_combinationsGF2_128_512_equal(1, 1);
    }
    else if (n <= 256) {
        set_256_512(bits);
        unsigned long long int  w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]) +
            popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w == w_searched) return true;
        weights[w]++;
        linear_combinationsGF2_256_512_equal(1, 1);

    }
    else {
        set_512(bits);
        if ((POPCNT == 2) || (POPCNT == -2)) {
            linear_combinations_512_equal(1, 1);
        }
        else {
            linear_combinations_512_equal_f(1, 1);
        }
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}


bool calculateWeightBytes_512_equal(dmat_type& bits, int n, int k, int m, int q, int d) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    equal_flag = true;

    register_elements = ((n - 1) / 512) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes_512(bits);
            linear_combinations_CH2_512_equal(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_49_512_equal(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_25_512_equal(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes_512(bits);
        linear_combinations_Bytes_512_equal(1, 1);
    }


    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}

bool calculateWeightCH3_512_equal(dynamic_mat_short& bits, int n, int k, int m, int d) {
    popcnt_detect();

    K = k;
    N = n;
    M = m;

    w_searched = d;
    equal_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    register_elements = (((n - 1) / 512) + 1);
    if ((POPCNT == 2) || (POPCNT == -2)) {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_equal(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_equal(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_equal(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_equal(1, 1);
            }
            else {
                linear_comb_recGF3_512_equal(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_equal(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_equal(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_equal(1, 1);
            }
            else {
                linear_combinations_GF9_512_equal(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_equal(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_equal(1, 1);
            }
            else {
                linear_combinations_GF27_512_equal(1, 1);
            }

        }
    }
    else {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_equal_f(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_equal_f(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_equal_f(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_equal_f(1, 1);
            }
            else {
                linear_comb_recGF3_512_equal_f(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_equal_f(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_equal_f(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_equal_f(1, 1);
            }
            else {
                linear_combinations_GF9_512_equal_f(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_equal_f(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_equal_f(1, 1);
            }
            else {
                linear_combinations_GF27_512_equal_f(1, 1);
            }

        }
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}


bool calculateWeightCH2_512_equal(dynamic_mat_short& bits, int n, int k, int m, int d) {

    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    register_elements = ((n - 1) / 512) + 1;

    w_searched = d;
    equal_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    setMatrixGF2_CF_512(bits);
    if (Q == 4) { // swich () case:
        if ((POPCNT == 2) || (POPCNT == -2)) {
            if (n <= 256) {
                linear_combinations_GF4_256_512_equal(1, 1);
            }
            else {
                linear_combinations_GF4_512_equal(1, 1);
            }
        }
        else {
            if (n <= 256) {
                linear_combinations_GF4_256_512_equal_f(1, 1);
            }
            else {
                linear_combinations_GF4_512_equal_f(1, 1);
            }
        }
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}




//-----------------------------calculateWeight_less_than--------------------------------------///
bool calculateWeightGF2_512_less_than(dynamic_mat_short& bits, int n, int k, int d) {
    popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    less_than_flag = true;

    register_elements = ((n - 1) / 512) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (n <= 64) {
        set_64_512(bits);
        unsigned long long int w = popcount(helper_GF2[0][1]);
        if (w < w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][2]);
        if (w < w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][3]);
        if (w < w_searched) return true;
        weights[w]++;
        linear_combinationsGF2_64_512_less(1, 1);
    }
    else if (n <= 128) {
        set_128_512(bits);

        unsigned long long int  w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        if (w < w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]);
        if (w < w_searched) return true;
        weights[w]++;

        w = popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w < w_searched) return true;
        weights[w]++;

        linear_combinationsGF2_128_512_less(1, 1);
    }
    else if (n <= 256) {
        set_256_512(bits);
        unsigned long long int  w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]) +
            popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w < w_searched) return true;
        weights[w]++;
        linear_combinationsGF2_256_512_less(1, 1);

    }
    else {
        set_512(bits);
        if ((POPCNT == 2) || (POPCNT == -2)) {
            linear_combinations_512_less(1, 1);
        }
        else {
            linear_combinations_512_less_f(1, 1);
        }
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}

bool calculateWeightBytes_512_less_than(dmat_type& bits, int n, int k, int m, int q, int d) {
    popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    less_than_flag = true;
    register_elements = ((n - 1) / 512) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes_512(bits);
            linear_combinations_CH2_512_less(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_49_512_less(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_25_512_less(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes_512(bits);
        linear_combinations_Bytes_512_less(1, 1);
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}


bool calculateWeightCH3_512_less_than(dynamic_mat_short& bits, int n, int k, int m, int d) {
    popcnt_detect();

    K = k;
    N = n;
    M = m;

    w_searched = d;
    less_than_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    register_elements = (((n - 1) / 512) + 1);
    if ((POPCNT == 2) || (POPCNT == -2)) {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_less(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_less(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_less(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_less(1, 1);
            }
            else {
                linear_comb_recGF3_512_less(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_less(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_less(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_less(1, 1);
            }
            else {
                linear_combinations_GF9_512_less(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_less(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_less(1, 1);
            }
            else {
                linear_combinations_GF27_512_less(1, 1);
            }

        }
    }
    else {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_less_f(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_less_f(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_less_f(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_less_f(1, 1);
            }
            else {
                linear_comb_recGF3_512_less_f(1, 1);
            }
        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_less_f(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_less_f(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_less_f(1, 1);
            }
            else {
                linear_combinations_GF9_512_less_f(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_less_f(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_less_f(1, 1);
            }
            else {
                linear_combinations_GF27_512_less_f(1, 1);
            }

        }

    }
    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}

bool calculateWeightCH2_512_less_than(dynamic_mat_short& bits, int n, int k, int m, int d) {

    popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    register_elements = ((n - 1) / 512) + 1;

    w_searched = d;
    less_than_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    setMatrixGF2_CF_512(bits);
    if (Q == 4) { // swich () case:
        if ((POPCNT == 2) || (POPCNT == -2)) {
            if (n <= 256) {
                linear_combinations_GF4_256_512_less(1, 1);
            }
            else {
                linear_combinations_GF4_512_less(1, 1);
            }
        }
        else {
            if (n <= 256) {
                linear_combinations_GF4_256_512_less_f(1, 1);
            }
            else {
                linear_combinations_GF4_512_less_f(1, 1);
            }
        }
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}


// --------------------------------- calculate the number of codewords with given weight and write then in a file ---------------------------//

unsigned long long int calculateNumberOfWordsCH3_512_equal(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {
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

    if ((POPCNT == 2) || (POPCNT == -2)) {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_equal_count(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_equal_count(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_equal_count(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_equal_count(1, 1);
            }
            else {
                linear_comb_recGF3_512_equal_count(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_equal_count(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_equal_count(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_equal_count(1, 1);
            }
            else {
                linear_combinations_GF9_512_equal_count(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_equal_count(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_equal_count(1, 1);
            }
            else {
                linear_combinations_GF27_512_equal_count(1, 1);
            }

        }
    }
    else {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_equal_count_f(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_equal_count_f(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_equal_count_f(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_equal_count_f(1, 1);
            }
            else {
                linear_comb_recGF3_512_equal_count_f(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_equal_count_f(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_equal_count_f(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_equal_count_f(1, 1);
            }
            else {
                linear_combinations_GF9_512_equal_count_f(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_equal_count_f(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_equal_count_f(1, 1);
            }
            else {
                linear_combinations_GF27_512_equal_count_f(1, 1);
            }

        }
    }

    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);

    return ct;

}

unsigned long long int  calculateNumberOfWordsBytes_512_equal(dmat_type& bits, int n, int k, int m, int q, int d, bool multiplicativeForm) {
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
            setRegistersBytes_512(bits);
            linear_combinations_CH2_512_equal_count(1, 1);
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
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_49_512_equal_count(1, 1);
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
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_25_512_equal_count(1, 1);
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
        setRegistersBytes_512(bits);
        linear_combinations_Bytes_512_equal_count(1, 1);
    }

    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}

unsigned long long int calculateNumberOfWordsCH2_512_equal(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {
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

    setMatrixGF2_CF_512(bits);
    if (Q == 4) { // swich () case:
        if ((POPCNT == 2) || (POPCNT == -2)) {
            if (n <= 256) {
                linear_combinations_GF4_256_512_equal_count(1, 1);
            }
            else {
                linear_combinations_GF4_512_equal_count(1, 1);
            }
        }
        else {
            if (n <= 256) {
                linear_combinations_GF4_256_512_equal_count_f(1, 1);
            }
            else {
                linear_combinations_GF4_512_equal_count_f(1, 1);
            }
        }
    }
    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}

unsigned long long int calculateNumberOfWordsGF2_512_equal(dynamic_mat_short& bits, int n, int k, int d, bool multiplicativeForm) {
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
    register_elements = ((n - 1) / 512) + 1;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }
    if (n <= 64) {
        set_64_512(bits);
        unsigned long long int w = popcount(helper_GF2[0][1]);
        if (w == w_searched) write_GF2_coset_64_512(0, 1);
        weights[w]++;

        w = popcount(helper_GF2[0][2]);
        if (w == w_searched) write_GF2_coset_64_512(0, 2);
        weights[w]++;

        w = popcount(helper_GF2[0][3]);
        if (w == w_searched) write_GF2_coset_64_512(0, 3);
        weights[w]++;
        linear_combinationsGF2_64_512_equal_count(1, 1);
    }
    else if (n <= 128) {
        set_128_512(bits);

        unsigned long long int  w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        if (w == w_searched) write_GF2_coset_128_512(0, 2);
        weights[w]++;

        w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]);
        if (w == w_searched) write_GF2_coset_128_512(0, 4);
        weights[w]++;

        w = popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w == w_searched) write_GF2_coset_128_512(0, 6);
        weights[w]++;

        linear_combinationsGF2_128_512_equal_count(1, 1);
    }
    else if (n <= 256) {
        set_256_512(bits);
        unsigned long long int  w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]) +
            popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w == w_searched) write_GF2_coset_512(0, 4);
        weights[w]++;
        linear_combinationsGF2_256_512_equal_count(1, 1);

    }
    else {
        set_512(bits);
        if ((POPCNT == 2) || (POPCNT == -2)) {
            linear_combinations_512_equal_count(1, 1);
        }
        else {
            linear_combinations_512_equal_count_f(1, 1);
        }
    }

    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}


// --------------------------------- END calculate the number of codewords with given weight and write then in a file ---------------------------//


// --------------------------------- calculate the number of codewords weight less than given value and write then in a file ---------------------------//

unsigned long long int calculateNumberOfWordsCH3_512_less_than(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {
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


    register_elements = (((N - 1) / 512) + 1);

    if ((POPCNT == 2) || (POPCNT == -2)) {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_less_count(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_less_count(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_less_count(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_less_count(1, 1);
            }
            else {
                linear_comb_recGF3_512_less_count(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_less_count(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_less_count(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_less_count(1, 1);
            }
            else {
                linear_combinations_GF9_512_less_count(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_less_count(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_less_count(1, 1);
            }
            else {
                linear_combinations_GF27_512_less_count(1, 1);
            }

        }
    }
    else {
        if (m == 1) {
            Q = 3;
            setMatrixGF3_512(bits);
            if (n <= 64) {
                linear_comb_recGF3_64_512_less_count_f(1, 1);
            }
            else if (n <= 128) {
                linear_comb_recGF3_128_512_less_count_f(1, 1);
            }
            else if (n <= 192) {
                linear_comb_recGF3_192_512_less_count_f(1, 1);
            }
            else if (n <= 256) {
                linear_comb_recGF3_256_512_less_count_f(1, 1);
            }
            else {
                linear_comb_recGF3_512_less_count_f(1, 1);
            }


        }
        else if (m == 2) {
            Q = 9;
            setMatrixGF9_512(bits);
            if (n <= 64) {
                linear_combinations_GF9_64_512_less_count_f(1, 1);
            }
            else if (n <= 128) {
                linear_combinations_GF9_128_512_less_count_f(1, 1);
            }
            else if (n <= 256) {
                linear_combinations_GF9_256_512_less_count_f(1, 1);
            }
            else {
                linear_combinations_GF9_512_equal_count_f(1, 1);
            }
        }
        else if (m == 3) {
            Q = 27;
            setMatrixGF27_512(bits);

            if (n <= 64) {
                linear_combinations_GF27_256_512_less_count_f(1, 1);

            }
            else if (n <= 256) {
                linear_combinations_GF27_256_512_less_count_f(1, 1);
            }
            else {
                linear_combinations_GF27_512_less_count_f(1, 1);
            }

        }
    }

    unsigned long long int ct = 0;
    unsigned long long int i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    fprintf(file, "\n\n");
    fclose(file);

    return ct;

}

unsigned long long int  calculateNumberOfWordsBytes_512_less_than(dmat_type& bits, int n, int k, int m, int q, int d, bool multiplicativeForm) {
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
                printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
            }
            setRegistersBytes_512(bits);
            linear_combinations_CH2_512_less_count(1, 1);
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
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_49_512_less_count(1, 1);
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
            setRegistersBytesCF_512(bits);
            linear_combinations_CF_25_512_less_count(1, 1);
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
        setRegistersBytes_512(bits);
        linear_combinations_Bytes_512_less_count(1, 1);
    }

    unsigned long long int ct = 0;
    unsigned long long int i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}

unsigned long long int calculateNumberOfWordsCH2_512_less_than(dynamic_mat_short& bits, int n, int k, int m, int d, bool multiplicativeForm) {
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

    setMatrixGF2_CF_512(bits);
    if (Q == 4) { // swich () case:
        if ((POPCNT == 2) || (POPCNT == -2)) {
            if (n <= 256) {
                linear_combinations_GF4_256_512_less_count(1, 1);
            }
            else {
                linear_combinations_GF4_512_less_count(1, 1);
            }
        }
        else {
            if (n <= 256) {
                linear_combinations_GF4_256_512_less_count_f(1, 1);
            }
            else {
                linear_combinations_GF4_512_less_count_f(1, 1);
            }
        }
    }
    unsigned long long int ct = 0;
    unsigned long long int i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}

unsigned long long int calculateNumberOfWordsGF2_512_less_than(dynamic_mat_short& bits, int n, int k, int d, bool multiplicativeForm) {
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
        printf("Cannot open file Result_codewords_CountLessThan.txt\n The codewords won't be writen!\n");
    }
    register_elements = ((n - 1) / 512) + 1;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }
    if (n <= 64) {
        set_64_512(bits);
        unsigned long long int w = popcount(helper_GF2[0][1]);
        if (w < w_searched) write_GF2_coset_64_512(0, 1);
        weights[w]++;

        w = popcount(helper_GF2[0][2]);
        if (w < w_searched) write_GF2_coset_64_512(0, 2);
        weights[w]++;

        w = popcount(helper_GF2[0][3]);
        if (w < w_searched) write_GF2_coset_64_512(0, 3);
        weights[w]++;
        linear_combinationsGF2_64_512_less_count(1, 1);
    }
    else if (n <= 128) {
        set_128_512(bits);

        unsigned long long int  w = popcount(helper_GF2[0][2]) + popcount(helper_GF2[0][3]);
        if (w < w_searched) write_GF2_coset_128_512(0, 2);
        weights[w]++;

        w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]);
        if (w < w_searched) write_GF2_coset_128_512(0, 4);
        weights[w]++;

        w = popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w < w_searched) write_GF2_coset_128_512(0, 6);
        weights[w]++;

        linear_combinationsGF2_128_512_less_count(1, 1);
    }
    else if (n <= 256) {
        set_256_512(bits);
        unsigned long long int  w = popcount(helper_GF2[0][4]) + popcount(helper_GF2[0][5]) +
            popcount(helper_GF2[0][6]) + popcount(helper_GF2[0][7]);
        if (w < w_searched) write_GF2_coset_512(0, 4);
        weights[w]++;
        linear_combinationsGF2_256_512_less_count(1, 1);

    }
    else {
        set_512(bits);
        if ((POPCNT == 2) || (POPCNT == -2)) {
            linear_combinations_512_less_count(1, 1);
        }
        else {
            linear_combinations_512_less_count_f(1, 1);
        }
    }

    unsigned long long int ct = 0;
    unsigned long long int i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}

// --------------------------------- END calculate the number of codewords weight less than given value and write then in a file ---------------------------//
#else

void calculateWeightCH3_512(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
}
void calculateWeightBytes_512(dmat_type& generatorMatrix_byte, int n, int k, int m, int q) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
}
void calculateWeightCH2_512(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
}
void calculateWeightGF2_512(dynamic_mat_short& generatorMatrix_bits, int n, int k) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
}
bool calculateWeightCH3_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightBytes_512_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightCH2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightGF2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightCH3_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int d) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightBytes_512_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int d){
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightCH2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int d) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
bool calculateWeightGF2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int d) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return false;
}
unsigned long long int calculateNumberOfWordsCH3_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsBytes_512_equal(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsCH2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsGF2_512_equal(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsCH3_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsBytes_512_less_than(dmat_type& generatorMatrix_byte, int n, int k, int m, int q, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsCH2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int m, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX512 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}
unsigned long long int calculateNumberOfWordsGF2_512_less_than(dynamic_mat_short& generatorMatrix_bits, int n, int k, int w, bool multiplicativeForm) {
    ERRORQ("ERROR:  AVX521 instructions were detected on CPU, but the corresponding compiler flag was not used! If working on Visual Studion, please change compiler flag at line 18 in CMakeLists.txt file in src directory to /arch:AVX512, or manually change the flags for project v1.3! Otherwise, please change the flag to  -march=skylake-avx512 in line 20 in the same in CMakeLists.txt file in src directory.");
    return 0;
}

#endif
