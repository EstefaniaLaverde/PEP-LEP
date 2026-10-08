#include <iostream>
#include <fstream>
#include <time.h>
#include "lib_neon.h"


#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#include <cpuid.h>
#elif defined(__GNUC__)
#include "arm_neon.h"
#endif

//function to check if CPU has popcount instruction
/*void popcnt_detect() {
#if defined(_MSC_VER)
    int abcd[4] = { 0,0,0,0 };
    __cpuid(abcd, 1);
    if (abcd[2] & (1 << 23)) {
        POPCNT = 1;
    }
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    unsigned int abcd[4] = { 0,0,0,0 };
    __get_cpuid(1, &abcd[0], &abcd[1], &abcd[2], &abcd[3]);
    if (abcd[2] & (1 << 23)) {
        POPCNT = 1;
    }
#endif
}*/


// function to choose an implementation of popcount based on compiler
// and if there is a popcount CPU instruction
// can be replaced with the specific implementation if the CPU is known
// for faster processing time

/*unsigned long long int popcount(unsigned long long int word){
    //8xf16
    uint8x8_t word_v = vld1_u8(&word);
    uint8x8_t r1 = vcnt_u8(word);
    uint8_t r[8] = {0,0,0,0,0,0,0,0};
    unsigned long long w=0;
    for(int i = 0; i<16; i++){
        w = w + r[i];
    }
    return w;
}*/

unsigned long long int popcount64(unsigned long long int word){
    uint8x8_t a_r = (uint8x8_t)word;
    uint8x8_t w_r =vcnt_u8(a_r);
    unsigned long long int w=vaddv_u8(w_r);
    return w;
}



unsigned long long int  popcount(uint8x16_t &word) {
    // 16x8???
    uint8x16_t r1 = vcntq_u8(word);
    unsigned long long int w=vaddvq_u8(r1);

    return w;
    
    /*if (POPCNT) {
#if defined(_MSC_VER)
        //return _popcnt64(word); // visual studio with clang
        return _mm_popcnt_u64(word); // visual studio msvc
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__) || defined (__clang__))
        return _popcnt64(word);
#endif
    }
    else {
        unsigned long long t_w;
        unsigned long long w_w;
        t_w = word - ((word >> 1) & 0x5555555555555555L);
        t_w = (t_w & 0x3333333333333333L) + ((t_w >> 2) & 0x3333333333333333L);
        t_w = ((t_w + (t_w >> 4)) & 0x0f0f0f0f0f0f0f0fL);
        w_w = (t_w * 0x0101010101010101L) >> 56;
        return (long long)(w_w);
    }*/
}


unsigned long long int weights[N_FIX * 8 + 1]; // for saving the weight spectrum of the code


// static arrays that are used to save the generator matrix into the registers
static union {
    //uint8x16_t reg128_matrix_GF2[K_GF2][N_GF2/2];
    unsigned long long int matrix_GF2[K_GF2][N_GF2]; // for GF2 and GF4; bitwise representation of the elements of the field with characteristic 2
   // uint8x16_t reg128_matrix_CH2[K_GF2][N_CH2 / 16];
    unsigned char matrix_CH2[K_GF2][N_CH2]; // for GF8, GF16, GF32, GF64; bytewise representation of the elements ofthe field with characteristic 2
    unsigned long long int matrix_CH3[K_CH3][N_CH3]; // for fields with characteristic 3; bitwise representation of the elements
   // uint8x16_t reg128_matrix_CH3[K_CH3][N_CH3/2];
    unsigned char matrix_p[K_P][N_P]; // for ohter finite fields (GF5, GF7, GF11, ..., GF25, ..., GF49, ...); bytewise representation of the elements
   // uint8x16_t reg128_matrix_p[K_P][N_P/16];
};

// array that save the current linear combination
// used to calculate naext linear combination
static union {
    //uint8x16_t reg128_helper_GF2[K_GF2][N_GF2 / 2];
    unsigned long long int helper_GF2[K_GF2][N_GF2];
    unsigned char helper_CH2[K_GF2][N_CH2];
    //uint8x16_t reg128_helper_CH2[K_GF2][N_CH2 / 16];
    unsigned long long int helper_CH3[K_CH3][N_CH3];
    //uint8x16_t reg128_helper_CH3[K_CH3][N_CH3/2];
    unsigned char helper_p[K_P][N_P];
    //uint8x16_t reg128_helper_p[K_P][N_P/16];
};


using namespace std;
/*
// global variables used in different functions
static int K = 0;
static int N = 0;
static int Q = 2;
static int M = 1; // for composite fields; gives the power of the characteristic of the field
static int w_searched = -1; // seves the searched weights
static bool less_than_flag = true; // flase if a word with weight less than the searched weight is found; the search stops
static bool equal_flag = true; // flase if a word with weight equal the searched weight is found; the search stops
FILE* file; // stream to a file if the codewords need to be saved for future calculation

static int register_elements = ((N_FIX*8 - 1) / 128) + 1; // gives the number of registers that will be used
static int Characteristic = 2;

static __m128i zero; // zero register; used for calculations with byte representation
static __m128i Q_reg_Bytes; // register containing 16 8-byte elemets representing the number of elements in the field; used for calculations with byte representation
bool form = false; // if true -> write the codewords as a power of primitive element

// transition sequences of Q-ary Grey code for field with different characteristics
// used to show which copy of the generator matrix is used for calculation of next codeword
static short int TransitionSequence64[64] = { 0, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 6,
                           1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1 };
static short int TransitionSequence25[25] = { 0,1,1,1,1, 2,1,1,1,1, 2,1,1,1,1, 2,1,1,1,1, 2,1,1,1,1 };
static short int TransitionSequence27[27] = { 0,1,1,2,1,1,2,1,1,3,1,1,2,1,1,2,1,1,3,1,1,2,1,1,2,1,1 };
static short int TransitionSequence49[49] = { 0,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1, 2,1,1,1,1,1,1 };
*/
static uint8x16_t zero; // zero register; used for calculations with byte representation
static uint8x16_t Q_reg_Bytes; // register containing 16 8-byte elemets representing the number of elements in the field; used for calculations with byte representation




// ----------------------------- BYTE REPRESENTATION ----------------------------------------- //
//sets registers for byte representation of the elements
void setRegistersBytes(dmat_type &bits) {
   // cout << "Set registers" << endl;
    zero = vdupq_n_u8(0);
    Q_reg_Bytes = vdupq_n_u8((uint8_t)Q);
    if (Q % 2 == 0) {

        for (int i = 0; i < K_GF2; i++) {
            for (int j = 0; j < N_CH2; j++) {
                matrix_CH2[i][j] = 0;
                helper_CH2[i][j] = 0;
            }
        }

        for (int i = 1; i <= (M*K); i++) {
            for (int j = 0; j < N; j++) {
                matrix_CH2[i][j] = bits.a[i-1][j];
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

        for (int i = 1; i <= (M*K); i++) {
            for (int j = 0; j < N; j++) {
                matrix_p[i][j] = bits.a[i-1][j];
                //cout <<(int) matrix_p[i][j] << " ";

            }
            //cout << endl;
        }
    }

}

//sets the registers for complimentary fields
//one register contains the coeficients in the polinomials for each power
//using 128-bit register
void setRegistersCF(dmat_type &bits) {
    //cout << "Set registers compl field" << endl;
    zero = vdupq_n_u8(0);
    Q_reg_Bytes = vdupq_n_u8((uint8_t)Characteristic);

    for (int row = 0; row < K_P; row++) {
        for (int col = 0; col < N_P; col++) {
            matrix_p[row][col] = 0;
            helper_p[row][col] = 0;
        }
    }
    int shift = (((N - 1) / 16) + 1);
    for (int row = 1; row <= (M*K); row++) {
        for (int col = 0; col < N; col++) {
            matrix_p[row][col] = bits.a[row - 1][col];
            matrix_p[row][col + 16 * shift] = bits.a[row - 1][col + N];
        }
    }
}

//adds two vectors in positions rec and i of the static matrix using 128 bit registers over prime fields

static inline void add(int rec, int i, int res) {
    for (int col = 0; col < N; col= col + 16) {
        //cout<<col<<"  "<<(int)helper_p[rec][col]<<"  "<<(int)matrix_p[i][col]<<endl;
        //int8_t show[16] ;
        
        int8x16_t v1 = vld1q_s8((int8_t*)&helper_p[rec][col]);
        //vst1q_s8 (show,v1); cout<<"matrix: \n";
        //for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
        //cout<<endl<<"helper: \n";
        int8x16_t v2 = vld1q_s8((int8_t*)&matrix_p[i][col]);
        //vst1q_s8 (show,v2);
       // for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
        //cout<<endl;
        
        int8x16_t r1 = vaddq_s8(v1, v2);
        //vst1q_s8(show,r1); cout<<"add: ";
        //for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
        //cout<<endl;
        
        int8x16_t r2 = vsubq_s8(r1, Q_reg_Bytes);
        //vst1q_s8 (show,r2); cout<<"sub: ";
        //for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
       // cout<<endl;
        
        int8x16_t m1 = vcltq_s8(r2, zero);
        //vst1q_s8 (show,m1); cout<<"mask1: ";
        //for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
        //cout<<endl;
        
        int8x16_t m2 = vcgeq_s8(r1,Q_reg_Bytes);
        //vst1q_s8 (show,m2); cout<<"mask2: ";
        //for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
        //cout<<endl;
        
        int8x16_t r3 = vandq_u8(r1, m1);
        //vst1q_s8 (show,r3); cout<<"r3: ";
        //for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
        //cout<<endl;
        
        int8x16_t r4 = vandq_u8(r2, m2);
       // vst1q_s8 (show,r4); cout<<"r4: ";
        //for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
        //cout<<endl;
        
        int8x16_t r5 = vorrq_u8(r3,r4);
       // vst1q_s8 (show,r5); cout<<"r5: ";
       // for(int j = 0; j<16; j++){cout<<(int)show[j]<<" ";}
       // cout<<endl;
        
        //cout<<endl<<endl;
        
        
        vst1q_u8((uint8_t*)&helper_p[res][col],r5);
        
    }
}




// calculates the weight for byte representation over prime fields
static inline unsigned long long int weight( int res) {
    uint8x16_t res_gt, v1;
    unsigned long long int w = 0;
    for (int col = 0; col < N; col+=16) {
        v1 = vld1q_u8((uint8_t*)&helper_p[res][col]);
        res_gt = vcgtq_s8(v1,zero);  // reg[i] > zero
        w = w + (popcount(res_gt)>>3);
    }
    return w;
}

//adds two vectors in positions rec and i of the static matrix with byte representation over complimentari fields
static inline void add_CF(int rec, int i, int res) {
    int shift = (((N - 1) / 16) + 1);
    for (int col = 0; col < N; col+=16) {
        //^0
        int8x16_t v1 = vld1q_s8((int8_t*)&helper_p[rec][col]);
        int8x16_t v2 = vld1q_s8((int8_t*)&matrix_p[i][col]);
        int8x16_t r1 = vaddq_s8(v1, v2);
        int8x16_t r2 = vsubq_s8(r1, Q_reg_Bytes);
        int8x16_t m1 = vcltq_s8(r2, zero);
        int8x16_t m2 = vcgeq_s8(r1,Q_reg_Bytes);
        int8x16_t r3 = vandq_u8(r1, m1);
        int8x16_t r4 = vandq_u8(r2, m2);
        int8x16_t r5 = vorrq_u8(r3,r4);
        vst1q_u8((uint8_t*)&helper_p[res][col],r5);
        
        //^1
        v1 = vld1q_s8((int8_t*)&helper_p[rec][col+shift*16]);
        v2 = vld1q_s8((int8_t*)&matrix_p[i][col+shift*16]);
        r1 = vaddq_s8(v1, v2);
        r2 = vsubq_s8(r1, Q_reg_Bytes);
        m1 = vcltq_s8(r2, zero);
        m2 = vcgeq_s8(r1,Q_reg_Bytes);
        r3 = vandq_u8(r1, m1);
        r4 = vandq_u8(r2, m2);
        r5 = vorrq_u8(r3,r4);
        vst1q_u8((uint8_t*)&helper_p[res][col+shift*16],r5);
        
    }
    
   /* __m128i res_add0, res_add1, res_sub0, res_sub1;
    unsigned long long int w = 0;
    int shift = (((N - 1) / 16) + 1);
    for (int col = 0; col < shift; col++) {
        // ^0
        res_add0 = _mm_add_epi8(reg128_helper_p[rec][col], reg128_matrix_p[i][col]);
        res_sub0 = _mm_sub_epi8(res_add0, Q_reg_Bytes);
        reg128_helper_p[res][col] = _mm_blendv_epi8(res_sub0, res_add0, res_sub0);

        // ^1
        res_add1 = _mm_add_epi8(reg128_helper_p[rec][col + shift], reg128_matrix_p[i][col + shift]);
        res_sub1 = _mm_sub_epi8(res_add1, Q_reg_Bytes);
        reg128_helper_p[res][col + shift] = _mm_blendv_epi8(res_sub1, res_add1, res_sub1);

    }
*/
}

//write the codeword save in helper array on position res in file for prime fields with byte representation
 void write_Bytes(int res) {
    if (file!=NULL) {
        for (int i = 0; i < N; i++) {
            int t = (int)helper_p[res][i];
            if (form) {
                write_multpl(t, file);
            }
            else {
                fprintf(file, "%d, ", t);
            }

        }
        fprintf(file, "\n");
    }
}


 void write_ByteCH2(int res) {
    if (file != NULL) {
        for (int i = 0; i < N; i++) {
            int t = (int)helper_CH2[res][i];//helper_p
            if (form) {
                write_multpl(t, file);
            }
            else {
                fprintf(file, "%d, ", t);
            }

        }
        fprintf(file, "\n");
    }
}

//write the codeword at position in file res for GF25 and GF49
void write_CF(int res) {
    if (file!=NULL) {
        int t = 0;
        int ch = 5;
        if (Q == 49) ch = 7;
        int shift = (((N - 1) / 16) + 1);
        for (int i = 0; i < N; i++) {
            t = 0;
            t =( ch * helper_p[res][i]) + helper_p[res][i + 16*shift];
            if (form) {
                write_multpl(t,  file);
            }
            else {
               fprintf(file, "%d, ", t);
            }
        }
        fprintf(file, "\n");
    }
}



// calculation the weight for composite fields with byte representation
static inline unsigned long long int weight_CF(int res) {
    uint8x16_t res_gt1, res_gt2, v1;
    int shift = (((N - 1) / 16) + 1);
    unsigned long long int w = 0;
    for (int col = 0; col < N; col+=16) {
        v1 = vld1q_u8((uint8_t*)&helper_p[res][col]);
        res_gt1 = vcgtq_s8(v1,zero);  // reg[i] > zero
        
        v1 = vld1q_u8((uint8_t*)&helper_p[res][col+16*shift]);
        res_gt2 = vcgtq_s8(v1,zero);  // reg[i] > zero
        
        v1 = vorrq_u8(res_gt1,res_gt2);
        w = w + (popcount(v1)>>3);
    }
    return w;
    
   /*
    //two_popcnt
    __m128i res_add0, res_add1;
    unsigned long long int w = 0;
    int shift = (((N - 1) / 16) + 1);
    for (int col = 0; col < shift; col++) {
        res_add0 = _mm_cmplt_epi8(zero, reg128_helper_p[res][col]);
        res_add1 = _mm_cmplt_epi8(zero, reg128_helper_p[res][col + shift]);


        __m128i weight_reg = _mm_or_si128(res_add0, res_add1);

        unsigned long long* t = (unsigned long long*) & weight_reg;
        w = w + ((popcount(t[0]) + popcount(t[1])) >> 3);
    }
    return w;
    */
    //one_popcnt
   /* __m128i r1, r2, r3, r4,r0, cmpl1, cmpl2;
    int shift = (((N - 1) / 16) + 1);
    __m128i h = _mm_set_epi8(2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1);
    unsigned long long int w = 0;
    for (int col = 0; col < (((N - 1) / 16) + 1); col++) {
        cmpl1 = _mm_cmpeq_epi8(zero, reg128_helper_p[res][col]);
        cmpl2 = _mm_cmpeq_epi8(zero, reg128_helper_p[res][col + shift]);
        r0 = _mm_and_si128(cmpl1, cmpl2);
        r2 = _mm_and_si128(r0, h);
        r3 = _mm_srli_si128(r2, 8);
        r4 = _mm_or_si128(r2, r3); //r1 = ...

        unsigned long long* t = (unsigned long long*) & r4;
        w = w + (16 - popcount(t[0]));
    }

    return w;
    
    */
}



//for fields over prime numbres
void linear_combinations_Bytes_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add(rec - 1, i, rec);
                }
                else {
                    add(rec, i, rec);
                }

                unsigned long long int w = weight(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_Bytes_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_Bytes_euqal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add(rec - 1, i, rec);
                }
                else {
                    add(rec, i, rec);
                }

                unsigned long long int w = weight(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_Bytes_euqal(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_Bytes_less_than_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add(rec - 1, i, rec);
                }
                else {
                    add(rec, i, rec);
                }

                unsigned long long int w = weight(rec);
                if (w < w_searched) {
                    write_Bytes(rec);
                }

                weights[w]++;

                if (rec < K) {
                    linear_combinations_Bytes_less_than_count(rec + 1, i + 1);
                }
            }
        }
    }



void linear_combinations_Bytes_euqal_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add(rec - 1, i, rec);
                }
                else {
                    add(rec, i, rec);
                }

                unsigned long long int w = weight(rec);
                if (w == w_searched) {
                    write_Bytes(rec);
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_Bytes_euqal_count(rec + 1, i + 1);
                }
            }
        }

}


void linear_combinations_Bytes(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add(rec - 1, i, rec);
                    unsigned long long int w = weight(rec);
                    weights[w]++;
                }
                else {
                    add(rec, i, rec);
                    unsigned long long int w = weight(rec);
                    weights[w]++;
                }

                if (rec < K) {
                    linear_combinations_Bytes(rec + 1, i + 1);
                }
            }
        }
}


//using transitional sequence for characteristic 7 - GF(49)
void linear_combinations_CF_49_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_49_less_than(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_CF_49_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_49_equal(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_CF_49_less_than_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w < w_searched) {
                    write_CF(rec);
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_49_less_than_count(rec + 1, i + 1);
                }
            }
        }
}

void linear_combinations_CF_49_equal_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w == w_searched) {
                    write_CF(rec);
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_49_equal_count(rec + 1, i + 1);
                }
            }
        }
}
void linear_combinations_CF_49(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                    unsigned long long int w = weight_CF(rec);
                    weights[w]++;
                }
                else {
                    short t = TransitionSequence49[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                    unsigned long long int w = weight_CF(rec);
                    weights[w]++;
                }
                if (rec < K) {
                    linear_combinations_CF_49(rec + 1, i + 1);
                }
            }
        }

}


//using transitional sequence for characteristic 5 and GF(25)
void linear_combinations_CF_25_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_25_less_than(rec + 1, i + 1);
                }
            }
        }
    }

}


void linear_combinations_CF_25_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_25_equal(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_CF_25_less_than_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w < w_searched) {
                    write_CF(rec);
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_25_less_than_count(rec + 1, i + 1);
                }
            }
        }
}


void linear_combinations_CF_25_equal_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                }

                unsigned long long int w = weight_CF(rec);
                if (w == w_searched)  {
                    write_CF(rec);
                }
                weights[w]++;

                if (rec < K) {
                    linear_combinations_CF_25_equal_count(rec + 1, i + 1);
                }
            }
        }
}

void linear_combinations_CF_25(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF(rec - 1, i, rec);
                    unsigned long long int w = weight_CF(rec);
                    weights[w]++;
                }
                else {
                    short t = TransitionSequence25[q1] - 1;
                    add_CF(rec, t * K + i, rec);
                    unsigned long long int w = weight_CF(rec);
                    weights[w]++;
                }
                if (rec < K) {
                    linear_combinations_CF_25(rec + 1, i + 1);
                }
            }
        }

}


// adds vectors over composite field with characteristic 2 and byte representation
static inline void add_CH2(int rec, int i, int res){
    uint8x16_t mat, help, result;

    for (int col = 0; col < N; col+=16) {
        mat = vld1q_u8((uint8_t*)&matrix_CH2[i][col]);
        help = vld1q_u8((uint8_t*)&helper_CH2[rec][col]);
        result = veorq_u64(mat,help);
        vst1q_u8((uint8_t*)&helper_CH2[res][col],result);
       // reg128_helper_CH2[res][col] = _mm_xor_si128(reg128_helper_CH2[rec][col], reg128_matrix_CH2[i][col]);
    }
}

// calculates the weights of vector over composite field with characteristic 2 and byte representation
static inline unsigned long long int weight_CH2(int res) {
    uint8x16_t res_gt, v1;
    unsigned long long int w = 0;
    for (int col = 0; col < N; col+=16) {
        v1 = vld1q_u8((uint8_t*)&helper_CH2[res][col]);
        res_gt = vcgtq_s8(v1,zero);  // reg[i] > zero
        w = w + (popcount(res_gt)>>3);
    }
    return w;
    /*
     unsigned long long int w = 0;
    for (int col = 0; col < ((N - 1) / 16) + 1; col++) {
        static union {
            __m128i res = _mm_setzero_si128();
            unsigned long long  res_64[2];
        };
        res = _mm_cmplt_epi8(zero, reg128_helper_CH2[result][col]);
        w = w + ((popcount(res_64[0]) + popcount(res_64[1])) >> 3);
    }
    return w;
     */
}



void linear_combinations_CH2_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2(rec, t * K + i, rec);
                }

                unsigned long long int weight = 0;
                weight = weight_CH2(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;

                if (rec < K) {
                    linear_combinations_CH2_less_than(rec + 1, i + 1);
                }
            }
        }
    }

}
void linear_combinations_CH2_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2(rec, t * K + i, rec);
                }

                unsigned long long int weight = 0;
                weight = weight_CH2(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;

                if (rec < K) {
                    linear_combinations_CH2_equal(rec + 1, i + 1);
                }
            }
        }
    }

}

void linear_combinations_CH2_less_than_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2(rec, t * K + i, rec);
                }

                unsigned long long int weight = 0;
                weight = weight_CH2(rec);
                if (weight < w_searched ) {
                    write_ByteCH2(rec);
                }
                weights[weight]++;

                if (rec < K) {
                    linear_combinations_CH2_less_than_count(rec + 1, i + 1);
                }
            }
        }
}
void linear_combinations_CH2_equal_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CH2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CH2(rec, t * K + i, rec);
                }

                unsigned long long int weight = 0;
                weight = weight_CH2(rec);
                if (weight == w_searched) {
                    write_ByteCH2(rec);
                }
                weights[weight]++;

                if (rec < K) {
                    linear_combinations_CH2_equal_count(rec + 1, i + 1);
                }
            }
        }
}

void linear_combinations_CH2(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    unsigned long long int weight = 0;
                    add_CH2(rec - 1, i, rec);
                    weight = weight_CH2(rec);
                    weights[weight]++;
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    unsigned long long int weight = 0;
                    add_CH2(rec, t * K + i, rec);
                    weight = weight_CH2(rec);
                    weights[weight]++;
                }
                if (rec < K) {
                    linear_combinations_CH2(rec + 1, i + 1);
                }
            }
        }
}

// ----------------------------- BYTE REPRESENTATION ----------------------------------------- //

// ----------------------------- Characteristic 3 --------------------------------------------//

// writing data in registers for GF3, bit representation
void setMatrixGF3(dynamic_mat_short &bits) {
    int c = (((N - 1) / 64) + 1);
    register_elements = (((N - 1) / 128) + 1);
    int bit1 = 0;

    // e.g. if we need 3 64-bit computer words for the given n, we will need 2 128-bit registers
    // writing scheme:
    // |    first bit of the representation    |    second bit of the representation   |
    // | 64 bits | 64 bits | 64 bits | ------- | 64 bits | 64 bits | 64 bits | ------- |
    // |      128 bits     |      128 bits     |      128 bits     |      128 bits     |

    if (c == 1 || c % 2 == 0) {
        bit1 = c;
    }
    else {
        bit1 = 2 * register_elements;
    }

    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3; el++) {
        matrix_CH3[0][el] = 0;
        helper_CH3[0][el] = (zero);//all 1 vector
    }

    for (int row = 1; row <= K; row++) {
        for (int i = 0; i < N_CH3; i++) {
            matrix_CH3[row][i] = (0);
            helper_CH3[row][i] = (0);
        }

        for (int el = 0; el < c; el++) {
            matrix_CH3[row][el] = bits.a[row - 1][el];
            matrix_CH3[row][el + bit1] = bits.a[row - 1][el + c];
        }
    }
}




// ---------------------------------------- functions for GF3 n<=64 ---------------------------------//
// addition function GF3 for n<=64
static inline void add_GF3_64(int j, int rec, int res) {
    uint64x2_t mat, help,rev_mat, rev_help,temp1, temp2, temp3;
    unsigned long long int rev[2]; 
    //printf("j = %d, rec = %d res = %d\n", j, rec, res);
    mat = vld1q_u64((uint64_t*)&matrix_CH3[j][0]); //printf("%llu  %llu\n",matrix_CH3[j][0],matrix_CH3[j][1]);
    help = vld1q_u64((uint64_t*)&helper_CH3[rec][0]); //printf("%llu  %llu\n",helper_CH3[rec][0],helper_CH3[rec][1]);

    rev[0] = matrix_CH3[j][1]; rev[1] = matrix_CH3[j][0]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_mat = vld1q_u64((uint64_t*)&rev);

    rev[0] = helper_CH3[rec][1]; rev[1] = helper_CH3[rec][0]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_help = vld1q_u64((uint64_t*)&rev);

    temp1 = veorq_u64(mat,help);
    temp2 = veorq_u64(rev_mat, rev_help);

    temp3 = veorq_u64(temp1,rev_mat);
    temp1 = vorrq_u64(temp3,temp2);

    vst1q_u64((uint64_t*)&helper_CH3[res][0],temp1);//printf("%llu  %llu\n\n\n",helper_CH3[res][0],helper_CH3[res][1]);


    /*
    __m128i xor_1 = _mm_setzero_si128();
    __m128i xor_2 = _mm_setzero_si128();
    __m128i xor_rev = _mm_setzero_si128();

    __m128i xor_rev2 = _mm_setzero_si128();

    xor_1 = _mm_xor_si128(reg128_matrix_CH3[j][0], reg128_helper_CH3[rec][0]);

    xor_rev2 = _mm_shuffle_epi32(reg128_matrix_CH3[j][0], 78);
    xor_2 = _mm_xor_si128(xor_1, xor_rev2);


    xor_rev = _mm_shuffle_epi32(xor_1, 78);
    // xor_rev = _mm_castpd_si128(_mm_permute_pd(
    //     _mm_castsi128_pd(xor_1), (int)1
    // ));
    reg128_helper_CH3[res][0] = _mm_or_si128(xor_2, xor_rev);
     */
}

//calculating weight for GF3 n<=64
static inline unsigned long long int weight_GF3_64(int res) {
    
    unsigned long long int w = 0;
    unsigned long long int w_and = helper_CH3[res][0] ^ helper_CH3[res][1];
    w = popcount64(w_and);
    return w;
     
}

// writing codeword into file GF3
 void write_GF3(int res) {
    if (file!=NULL) {
        int c = (((N - 1) / 64) + 1);
        int bit1 = 0;
        if (c == 1 || c % 2 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 2 * register_elements;
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
                    return; }
                bool first = helper_CH3[res][i] & (one << (63 - shift));
                bool second = helper_CH3[res][i+bit1] & (one << (63 - shift));
                if (first && second) { fprintf(file, "%d", 0);  }
                else if (first) { fprintf(file, "%d", 1); }
                else if (second) { fprintf(file, "%d", 2);  }
                else { cout << "EROR in witing in file for GF3 - element is 00!\n\n\n"; return; }
            }
        }
        fprintf(file, "\n");
    }
}

void linear_comb_recGF3_64_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64(j, rec - 2 + q1, rec);
                int w = weight_GF3_64(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_64_less_than(rec + 1, j + 1);
                }
            }

        }
    }

}

void linear_comb_recGF3_64_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64(j, rec - 2 + q1, rec);
                int w = weight_GF3_64(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_64_equal(rec + 1, j + 1);
                }
            }

        }
    }

}



void linear_comb_recGF3_64_less_than_count(int rec, int h) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64(j, rec - 2 + q1, rec);
                int w = weight_GF3_64(rec);
                if (w < w_searched){
                    write_GF3(rec);
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_64_less_than_count(rec + 1, j + 1);
                }
            }

        }
}

void linear_comb_recGF3_64_equal_count(int rec, int h) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64(j, rec - 2 + q1, rec);
                int w = weight_GF3_64(rec);
                if (w == w_searched) {
                    write_GF3(rec);
                }

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_64_equal_count(rec + 1, j + 1);
                }
            }

        }
}



void linear_comb_recGF3_64(int rec, int h) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_64(j, rec - 2 + q1, rec);
                int w = weight_GF3_64(rec);

                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_64(rec + 1, j + 1);
                }
            }

        }
}


// ----------------------------------- functions for GF3 n>64 ----------------------------//
//addition for GF3 n>64
static inline void add_GF3_SSE(int j, int rec, int res) {
    uint64x2_t mat1, help1, mat2, help2, t1,t2,t3,t4,t5,t6;

    for(int col = 0; col<2*register_elements; col= col + 2){
        mat1 = vld1q_u64((uint64_t*)&matrix_CH3[j][col]);
        help1 = vld1q_u64((uint64_t*)&helper_CH3[rec][col]);
        t1 = veorq_u64(mat1,help1);

        mat2 = vld1q_u64((uint64_t*)&matrix_CH3[j][col+2*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[rec][col+2*register_elements]);
        t2 = veorq_u64(mat2,help2);

        t3 = veorq_u64(t1,mat2);
        t4 = veorq_u64(t2,mat1);

        t5 = vorrq_u64(t3,t2);
        t6 = vorrq_u64(t1,t4);

        vst1q_u64((uint64_t*)&helper_CH3[res][col],t5);
        vst1q_u64((uint64_t*)&helper_CH3[res][col + 2*register_elements],t6);
    }

    /*
    __m128i xor_1[2];
    __m128i xor_2[2];

    for (int el = 0; el < register_elements; el++) {
       1) xor_1[0] = _mm_xor_si128(reg128_matrix_CH3[j][el], reg128_helper_CH3[rec][el]);
       2) xor_1[1] = _mm_xor_si128(reg128_matrix_CH3[j][el + register_elements], reg128_helper_CH3[rec][el + register_elements]);

       3) xor_2[0] = _mm_xor_si128(xor_1[0], reg128_matrix_CH3[j][el + register_elements]);
       4) xor_2[1] = _mm_xor_si128(xor_1[1], reg128_matrix_CH3[j][el]);

        reg128_helper_CH3[res][el] = _mm_or_si128(xor_2[0], xor_1[1]);
        reg128_helper_CH3[res][el + register_elements] = _mm_or_si128(xor_2[1], xor_1[0]);
    }
*/
}

// calculating the weight for GF3 n>64
static inline unsigned long long int weight_GF3_SSE(int res) {
    unsigned long long int w64[2]={0,0};
    uint64x2_t help1, help2, result;
    unsigned long long int w = 0;

    for(int i = 0 ; i<2*register_elements; i+=2){
        help1 = vld1q_u64((uint64_t*)&helper_CH3[res][i]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[res][i+2*register_elements]);
        result = veorq_u64(help1,help2);
        vst1q_u64((uint64_t*)&w64,result);
        w = w + popcount64(w64[0]) + popcount64(w64[1]);
    }
    return w;
    /*unsigned long long int w = 0;
    static union {
        __m128i w_xor = _mm_setzero_si128();
        unsigned long long  w_xor64[2];
    };
    for (int el = 0; el < register_elements; el++) {

        w_xor = _mm_xor_si128(reg128_helper_CH3[res][el], reg128_helper_CH3[res][el + register_elements]);
        w = w + popcount(w_xor64[0]) + popcount(w_xor64[1]);
    }
    return w;*/
}

void linear_comb_recGF3_SSE_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_SSE(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_SSE(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_SSE_less_than(rec + 1, j + 1);
                }
            }

        }
    }

}

void linear_comb_recGF3_SSE_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_SSE(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_SSE(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_SSE_equal(rec + 1, j + 1);
                }
            }

        }
    }

}

void linear_comb_recGF3_SSE_less_than_count(int rec, int h) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_SSE(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_SSE(rec);
                if (w < w_searched) {
                    write_GF3(rec);
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_SSE_less_than_count(rec + 1, j + 1);
                }
            }

        }
}

void linear_comb_recGF3_SSE_equal_count(int rec, int h) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_SSE(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_SSE(rec);
                if (w == w_searched ) {
                    write_GF3(rec);
                }
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_SSE_equal_count(rec + 1, j + 1);
                }
            }

        }
}

void linear_comb_recGF3_SSE(int rec, int h) {
        int qf = 2;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                add_GF3_SSE(j, rec - 2 + q1, rec);
                unsigned long long int  w = weight_GF3_SSE(rec);
                weights[w]++;

                if (rec < K) {
                    linear_comb_recGF3_SSE(rec + 1, j + 1);
                }
            }

        }
}

// ----------------------------------- functions for GF9 n<=64 ----------------------------------//


//writing data in the registers GF9; bit representation
void setMatrixGF9v2(dynamic_mat_short &bits) {
    int c = (((N - 1) / 64) + 1);

    int bit1 = 0;
    if (c == 1 || c % 2 == 0) {
        bit1 = c;
    }
    else {
        bit1 = 2 * register_elements;
    }

// writing scheme:
//
// |    first bit for the coef. with ^0    |    second bit for the coef. with ^0   |    first bit for the coef. with ^1    |    second bit for the coef. with ^1   |
// | 64 bits | 64 bits | 64 bits | ------- | 64 bits | 64 bits | 64 bits | ------- | 64 bits | 64 bits | 64 bits | ------- | 64 bits | 64 bits | 64 bits | ------- |
// |      128 bits     |      128 bits     |      128 bits     |      128 bits     |      128 bits     |      128 bits     |      128 bits     |      128 bits     |

    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    for (int el = 0; el < N_CH3; el++) {
        matrix_CH3[0][el] = (0);
        helper_CH3[0][el] = (zero);//all 1 vector
    }

    for (int row = 1; row <= 2 * K; row++) {
        for (int i = 0; i < N_CH3; i++) {
            matrix_CH3[row][i] = (0);
            helper_CH3[row][i] = (0);
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




// addition for GF9 n<=64
static inline void addGF9_64_SSEv2(int j, int rec, int res) {
    uint64x2_t mat, help,rev_mat, rev_help,temp1, temp2, temp3;
    unsigned long long int rev[2]; 
    //printf("j = %d, rec = %d res = %d\n", j, rec, res);
    mat = vld1q_u64((uint64_t*)&matrix_CH3[j][0]); //printf("%llu  %llu\n",matrix_CH3[j][0],matrix_CH3[j][1]);
    help = vld1q_u64((uint64_t*)&helper_CH3[rec][0]); //printf("%llu  %llu\n",helper_CH3[rec][0],helper_CH3[rec][1]);

    rev[0] = matrix_CH3[j][1]; rev[1] = matrix_CH3[j][0]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_mat = vld1q_u64((uint64_t*)&rev);

    rev[0] = helper_CH3[rec][1]; rev[1] = helper_CH3[rec][0]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_help = vld1q_u64((uint64_t*)&rev);

    temp1 = veorq_u64(mat,help);
    temp2 = veorq_u64(rev_mat, rev_help);
    temp3 = veorq_u64(temp1,rev_mat);
    temp1 = vorrq_u64(temp3,temp2);

    vst1q_u64((uint64_t*)&helper_CH3[res][0],temp1);

    mat = vld1q_u64((uint64_t*)&matrix_CH3[j][2]); //printf("%llu  %llu\n",matrix_CH3[j][0],matrix_CH3[j][1]);
    help = vld1q_u64((uint64_t*)&helper_CH3[rec][2]); //printf("%llu  %llu\n",helper_CH3[rec][0],helper_CH3[rec][1]);

    rev[0] = matrix_CH3[j][3]; rev[1] = matrix_CH3[j][2]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_mat = vld1q_u64((uint64_t*)&rev);

    rev[0] = helper_CH3[rec][3]; rev[1] = helper_CH3[rec][2]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_help = vld1q_u64((uint64_t*)&rev);

    temp1 = veorq_u64(mat,help);
    temp2 = veorq_u64(rev_mat, rev_help);
    temp3 = veorq_u64(temp1,rev_mat);
    temp1 = vorrq_u64(temp3,temp2);

    vst1q_u64((uint64_t*)&helper_CH3[res][2],temp1);
    
}

// calculating the weight for GF9 n<=64
static inline unsigned long long int weight_GF9_64_SSEv2(int res) {
    unsigned long long  element1 = 0, element2 = 0;
    element1 = helper_CH3[res][0] ^ helper_CH3[res][1];
    element2 = helper_CH3[res][2] ^ helper_CH3[res][3];
    unsigned long long int weight = 0;
    weight = popcount64(element1 | element2);

    return weight;
}

// writing the codeword into file for GF9
 void write_GF9(int res) {
    if (file!=NULL) {
        int c = (((N - 1) / 64) + 1);

        int bit1 = 0;
        if (c == 1 || c % 2 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 2 * register_elements;
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
                unsigned long long int first = helper_CH3[res][i] & (one << (63 - shift));
                unsigned long long int second = helper_CH3[res][i + bit1] & (one << (63 - shift));
                if (first != 0 && second != 0) { temp = 0; }
                else if (first != 0) { temp = 1; }
                else if (second != 0) { temp = 2; }
                else { cout << "EROR in witing in file for GF9 - element is 00!\n\n\n"; return; }
                result = result + temp;
                // cout << "res = " << res << endl;
                first = helper_CH3[res][i + 2 * bit1] & (one << (63 - shift));
                second = helper_CH3[res][i + 3 * bit1] & (one << (63 - shift));
                if (first != 0 && second != 0) { temp = 0; }
                else if (first != 0) { temp = 1; }
                else if (second != 0) { temp = 2; }
                else { cout << "EROR in witing in file for GF9 - element is 00!\n\n\n"; return; }
                result = result + 3 * temp;

                if (form) {
                    write_multpl(result,  file);
                }
                else {
                    fprintf(file, "%d, ", result);
                }
            }
        }
            fprintf(file, "\n");
    }
}
void linear_combinations_GF9_64_SSEv2_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_64_SSEv2(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_64_SSEv2_less_than(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_64_SSEv2_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_64_SSEv2(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_64_SSEv2_equal(rec + 1, j + 1);
                }
            }
        }
    }

}



void linear_combinations_GF9_64_SSEv2_less_than_count(int rec, int h) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_64_SSEv2(rec);
                if (weight < w_searched) {
                    write_GF9(rec);
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_64_SSEv2_less_than_count(rec + 1, j + 1);
                }
            }
        }
}

void linear_combinations_GF9_64_SSEv2_equal_count(int rec, int h) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_64_SSEv2(rec);
                if (weight == w_searched ) {
                    write_GF9(rec);
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_64_SSEv2_equal_count(rec + 1, j + 1);
                }
            }
        }
}

void linear_combinations_GF9_64_SSEv2(int rec, int h) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 8; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long weight = weight_GF9_64_SSEv2(rec);
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF9_64_SSEv2(rec + 1, j + 1);
                }
            }
        }
}
 // ------------------------------------- functions for GF9 n>64 ----------------------------------------------------//
static inline void addGF9_SSEv2(int j, int rec, int res) {
    uint64x2_t mat1, help1, mat2, help2, t1,t2,t3,t4,t5,t6;

    for(int col = 0; col<2*register_elements; col= col + 2){
        //^0
        mat1 = vld1q_u64((uint64_t*)&matrix_CH3[j][col]);
        help1 = vld1q_u64((uint64_t*)&helper_CH3[rec][col]);
        t1 = veorq_u64(mat1,help1);

        mat2 = vld1q_u64((uint64_t*)&matrix_CH3[j][col+2*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[rec][col+2*register_elements]);
        t2 = veorq_u64(mat2,help2);

        t3 = veorq_u64(t1,mat2);
        t4 = veorq_u64(t2,mat1);
        t5 = vorrq_u64(t3,t2);
        t6 = vorrq_u64(t1,t4);

        vst1q_u64((uint64_t*)&helper_CH3[res][col],t5);
        vst1q_u64((uint64_t*)&helper_CH3[res][col + 2*register_elements],t6);

        //^1
        mat1 = vld1q_u64((uint64_t*)&matrix_CH3[j][col + 4*register_elements]);
        help1 = vld1q_u64((uint64_t*)&helper_CH3[rec][col + 4*register_elements]);
        t1 = veorq_u64(mat1,help1);

        mat2 = vld1q_u64((uint64_t*)&matrix_CH3[j][col + 6*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[rec][col + 6*register_elements]);
        t2 = veorq_u64(mat2,help2);

        t3 = veorq_u64(t1,mat2);
        t4 = veorq_u64(t2,mat1);
        t5 = vorrq_u64(t3,t2);
        t6 = vorrq_u64(t1,t4);

        vst1q_u64((uint64_t*)&helper_CH3[res][col + 4*register_elements],t5);
        vst1q_u64((uint64_t*)&helper_CH3[res][col + 6*register_elements],t6);
    }

    /*__m128i xor_1[2];// = _mm_setzero_si128();
    __m128i xor_2[2];// = _mm_setzero_si128();

    xor_1[0] = _mm_setzero_si128();
    xor_1[1] = _mm_setzero_si128();
    xor_2[0] = _mm_setzero_si128();
    xor_2[1] = _mm_setzero_si128();
    for (int i = 0; i < register_elements; i++) {
        xor_1[0] = _mm_xor_si128(reg128_matrix_CH3[j][i], reg128_helper_CH3[rec][i]);
        xor_1[1] = _mm_xor_si128(reg128_matrix_CH3[j][i + register_elements], reg128_helper_CH3[rec][i + register_elements]);

        xor_2[0] = _mm_xor_si128(xor_1[0], reg128_matrix_CH3[j][i + register_elements]);
        xor_2[1] = _mm_xor_si128(xor_1[1], reg128_matrix_CH3[j][i]);

        reg128_helper_CH3[res][i] = _mm_or_si128(xor_2[0], xor_1[1]);
        reg128_helper_CH3[res][i + register_elements] = _mm_or_si128(xor_2[1], xor_1[0]);

        xor_1[0] = _mm_xor_si128(reg128_matrix_CH3[j][i + 2 * register_elements], reg128_helper_CH3[rec][i + 2 * register_elements]);
        xor_1[1] = _mm_xor_si128(reg128_matrix_CH3[j][i + 3 * register_elements], reg128_helper_CH3[rec][i + 3 * register_elements]);

        xor_2[0] = _mm_xor_si128(xor_1[0], reg128_matrix_CH3[j][i + 3 * register_elements]);
        xor_2[1] = _mm_xor_si128(xor_1[1], reg128_matrix_CH3[j][i + 2 * register_elements]);

        reg128_helper_CH3[res][i + 2 * register_elements] = _mm_or_si128(xor_2[0], xor_1[1]);
        reg128_helper_CH3[res][i + 3 * register_elements] = _mm_or_si128(xor_2[1], xor_1[0]);
    }*/
}

static inline unsigned long long int weight_GF9_SSEv2(int res) {
    unsigned long long int w64[2]={0,0};
    uint64x2_t help1, help2, element1, element2,result;
    unsigned long long int w = 0;

    for(int i = 0 ; i<2*register_elements; i+=2){
        help1 = vld1q_u64((uint64_t*)&helper_CH3[res][i]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[res][i+2*register_elements]);
        element1 = veorq_u64(help1,help2);
        help1 = vld1q_u64((uint64_t*)&helper_CH3[res][i+4*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[res][i+6*register_elements]);
        element2 = veorq_u64(help1,help2);
        result = vorrq_u64(element1,element2);
        vst1q_u64((uint64_t*)&w64,result);
        w = w + popcount64(w64[0]) + popcount64(w64[1]);
    }
    return w;
    /*
    for (int i = 0; i < register_elements; i++) {
        element1 = _mm_xor_si128(reg128_helper_CH3[res][i], reg128_helper_CH3[res][i + register_elements]);
        element2 = _mm_xor_si128(reg128_helper_CH3[res][i + 2 * register_elements], reg128_helper_CH3[res][i + 3 * register_elements]);
        temp = _mm_or_si128(element1, element2);
        count = count + popcount(temp64[0]) + popcount(temp64[1]);
    }
    return count;*/
}


void linear_combinations_GF9_SSEv2_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 8;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_SSEv2(rec);
                if (w < w_searched ) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_SSEv2_less_than(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF9_SSEv2_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 8;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_SSEv2(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_SSEv2_equal(rec + 1, j + 1);
                }
            }
        }
    }
}



void linear_combinations_GF9_SSEv2_less_than_count(int rec, int h) {
        int qf = 8;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_SSEv2(rec);
                if (w < w_searched ) {
                    write_GF9(rec);
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_SSEv2_less_than_count(rec + 1, j + 1);
                }
            }
        }
}

void linear_combinations_GF9_SSEv2_equal_count(int rec, int h) {
        int qf = 8;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_SSEv2(rec);
                if (w == w_searched) {
                    write_GF9(rec);
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_SSEv2_equal_count(rec + 1, j + 1);
                }
            }
        }
}

void linear_combinations_GF9_SSEv2(int rec, int h) {
        int qf = 8;
        if (h == 1) { qf = 1; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF9_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF9_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF9_SSEv2(rec);
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF9_SSEv2(rec + 1, j + 1);
                }
            }
        }

}


// writing data for GF27, bitwise representation
void setMatrixGF27v2(dynamic_mat_short &bits) {

    int c = (((N - 1) / 64) + 1);
    register_elements = ((N - 1) / 128) + 1;
    int bit1 = 0;
    if (c == 1 || c % 2 == 0) {
        bit1 = c;
    }
    else {
        bit1 = 2 * register_elements;
    }

    unsigned long long int zero = 18446744073709551615U;//(1 << 64) - 1;
    
    for (int el = 0; el < N_CH3; el++) {
        matrix_CH3[0][el] = (0);;
        helper_CH3[0][el] = (zero);//all 1 vector
    }

    for (int row = 1; row <= 3 * K; row++) {
        for (int i = 0; i < N_CH3; i++) {
            matrix_CH3[row][i] = (0);
            helper_CH3[row][i] = (0);
           // reg128_reverse[row][i] = vdupq_n_u8(0);
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

//------------------------------------- functions for GF27 n<=64 ------------------------------------//
static inline void addGF27_64_SSEv2(int j, int rec, int res) {
    uint64x2_t mat, help,rev_mat, rev_help,temp1, temp2, temp3;
    unsigned long long int rev[2]; 

    //element I
    mat = vld1q_u64((uint64_t*)&matrix_CH3[j][0]); //printf("%llu  %llu\n",matrix_CH3[j][0],matrix_CH3[j][1]);
    help = vld1q_u64((uint64_t*)&helper_CH3[rec][0]); //printf("%llu  %llu\n",helper_CH3[rec][0],helper_CH3[rec][1]);

    rev[0] = matrix_CH3[j][1]; rev[1] = matrix_CH3[j][0]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_mat = vld1q_u64((uint64_t*)&rev);

    rev[0] = helper_CH3[rec][1]; rev[1] = helper_CH3[rec][0]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_help = vld1q_u64((uint64_t*)&rev);

    temp1 = veorq_u64(mat,help);
    temp2 = veorq_u64(rev_mat, rev_help);
    temp3 = veorq_u64(temp1,rev_mat);
    temp1 = vorrq_u64(temp3,temp2);

    vst1q_u64((uint64_t*)&helper_CH3[res][0],temp1);

    //element II
    mat = vld1q_u64((uint64_t*)&matrix_CH3[j][2]); //printf("%llu  %llu\n",matrix_CH3[j][0],matrix_CH3[j][1]);
    help = vld1q_u64((uint64_t*)&helper_CH3[rec][2]); //printf("%llu  %llu\n",helper_CH3[rec][0],helper_CH3[rec][1]);

    rev[0] = matrix_CH3[j][3]; rev[1] = matrix_CH3[j][2]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_mat = vld1q_u64((uint64_t*)&rev);

    rev[0] = helper_CH3[rec][3]; rev[1] = helper_CH3[rec][2]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_help = vld1q_u64((uint64_t*)&rev);

    temp1 = veorq_u64(mat,help);
    temp2 = veorq_u64(rev_mat, rev_help);
    temp3 = veorq_u64(temp1,rev_mat);
    temp1 = vorrq_u64(temp3,temp2);

    vst1q_u64((uint64_t*)&helper_CH3[res][2],temp1);

    //elemet III
        mat = vld1q_u64((uint64_t*)&matrix_CH3[j][4]); //printf("%llu  %llu\n",matrix_CH3[j][0],matrix_CH3[j][1]);
    help = vld1q_u64((uint64_t*)&helper_CH3[rec][4]); //printf("%llu  %llu\n",helper_CH3[rec][0],helper_CH3[rec][1]);

    rev[0] = matrix_CH3[j][5]; rev[1] = matrix_CH3[j][4]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_mat = vld1q_u64((uint64_t*)&rev);

    rev[0] = helper_CH3[rec][5]; rev[1] = helper_CH3[rec][4]; //printf("%llu  %llu\n",rev[0], rev[1]);
    rev_help = vld1q_u64((uint64_t*)&rev);

    temp1 = veorq_u64(mat,help);
    temp2 = veorq_u64(rev_mat, rev_help);
    temp3 = veorq_u64(temp1,rev_mat);
    temp1 = vorrq_u64(temp3,temp2);

    vst1q_u64((uint64_t*)&helper_CH3[res][4],temp1);

}


static inline unsigned long long int weight_GF27_64_SSEv2(int res) {
    unsigned long long  element1 = 0, element2 = 0, element3 = 0;
    element1 = helper_CH3[res][0] ^ helper_CH3[res][1];
    element2 = helper_CH3[res][2] ^ helper_CH3[res][3];
    element3 = helper_CH3[res][4] ^ helper_CH3[res][5];

    unsigned long long int count = 0;
    count = popcount64(element1 | element2 | element3);
    return count;
     
}
void write_GF27(int res) {
    if (file!=NULL) {
        int c = (((N - 1) / 64) + 1);

        int bit1 = 0;
        if (c == 1 || c % 2 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 2 * register_elements;
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
                if (i * 64 + shift > (N - 1)) { fprintf(file, "\n"); return; }
                bool first = helper_CH3[res][i] & (one << (63 - shift));
                bool second = helper_CH3[res][i + bit1] & (one << (63 - shift));
                if (first && second) { temp = 0; }
                else if (first) { temp = 1; }
                else if (second) { temp = 2; }
                else { cout << "EROR in witing in file for GF27 - element is 00!\n\n\n"; return; }
                result = result + temp;

                first = helper_CH3[res][i + 2 * bit1] & (one << (63 - shift));
                second = helper_CH3[res][i + 3 * bit1] & (one << (63 - shift));
                if (first && second) { temp = 0; }
                else if (first) { temp = 1; }
                else if (second) { temp = 2; }
                else { cout << "EROR in witing in file for GF27 - element is 00!\n\n\n"; return; }
                result = result + 3 * temp;

                first = helper_CH3[res][i + 4 *bit1] & (one << (63 - shift));
                second = helper_CH3[res][i + 5 * bit1] & (one << (63 - shift));
                if (first && second) { temp = 0; }
                else if (first) { temp = 1; }
                else if (second) { temp = 2; }
                else { cout << "EROR in witing in file for GF27 - element is 00!\n\n\n"; return; }
                result = result + 9 * temp;
                if (form) {
                    write_multpl(result,  file);
                }
                else {
                    fprintf(file, "%d, ", result);
                }
            }
        }
        fprintf(file, "\n");
    }
}

void linear_combinations_GF27_64_SSEv2_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_64_SSEv2(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_64_SSEv2_less_than(rec + 1, j + 1);
                }
            }
        }
    }


}


void linear_combinations_GF27_64_SSEv2_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_64_SSEv2(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_64_SSEv2_equal(rec + 1, j + 1);
                }
            }
        }
    }
}


void linear_combinations_GF27_64_SSEv2_less_than_count(int rec, int h) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_64_SSEv2(rec);
                if ( w < w_searched) {
                    write_GF27(rec);
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_64_SSEv2_less_than_count(rec + 1, j + 1);
                }
            }
        }
}


void linear_combinations_GF27_64_SSEv2_equal_count(int rec, int h) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_64_SSEv2(rec);
                if ( w == w_searched) {
                    write_GF27(rec);
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_64_SSEv2_equal_count(rec + 1, j + 1);
                }
            }
        }
}

void linear_combinations_GF27_64_SSEv2(int rec, int h) {

        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_64_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_64_SSEv2(t * K + j, rec, rec);
                }
                unsigned long long int w = weight_GF27_64_SSEv2(rec);
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_64_SSEv2(rec + 1, j + 1);
                }
            }
        }
}

// ------------------------------------ functions for GF27 n>64 ------------------------ //
static inline void addGF27_SSEv2(int j, int rec, int res) {
    uint64x2_t mat1, help1, mat2, help2, t1,t2,t3,t4,t5,t6;

    for(int col = 0; col<2*register_elements; col= col + 2){
        //^0
        mat1 = vld1q_u64((uint64_t*)&matrix_CH3[j][col]);
        help1 = vld1q_u64((uint64_t*)&helper_CH3[rec][col]);
        t1 = veorq_u64(mat1,help1);

        mat2 = vld1q_u64((uint64_t*)&matrix_CH3[j][col+2*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[rec][col+2*register_elements]);
        t2 = veorq_u64(mat2,help2);

        t3 = veorq_u64(t1,mat2);
        t4 = veorq_u64(t2,mat1);
        t5 = vorrq_u64(t3,t2);
        t6 = vorrq_u64(t1,t4);

        vst1q_u64((uint64_t*)&helper_CH3[res][col],t5);
        vst1q_u64((uint64_t*)&helper_CH3[res][col + 2*register_elements],t6);

        //^1
        mat1 = vld1q_u64((uint64_t*)&matrix_CH3[j][col + 4*register_elements]);
        help1 = vld1q_u64((uint64_t*)&helper_CH3[rec][col + 4*register_elements]);
        t1 = veorq_u64(mat1,help1);

        mat2 = vld1q_u64((uint64_t*)&matrix_CH3[j][col + 6*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[rec][col + 6*register_elements]);
        t2 = veorq_u64(mat2,help2);

        t3 = veorq_u64(t1,mat2);
        t4 = veorq_u64(t2,mat1);
        t5 = vorrq_u64(t3,t2);
        t6 = vorrq_u64(t1,t4);

        vst1q_u64((uint64_t*)&helper_CH3[res][col + 4*register_elements],t5);
        vst1q_u64((uint64_t*)&helper_CH3[res][col + 6*register_elements],t6);

        //^2
        mat1 = vld1q_u64((uint64_t*)&matrix_CH3[j][col + 8*register_elements]);
        help1 = vld1q_u64((uint64_t*)&helper_CH3[rec][col + 8*register_elements]);
        t1 = veorq_u64(mat1,help1);

        mat2 = vld1q_u64((uint64_t*)&matrix_CH3[j][col + 10*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[rec][col + 10*register_elements]);
        t2 = veorq_u64(mat2,help2);

        t3 = veorq_u64(t1,mat2);
        t4 = veorq_u64(t2,mat1);
        t5 = vorrq_u64(t3,t2);
        t6 = vorrq_u64(t1,t4);

        vst1q_u64((uint64_t*)&helper_CH3[res][col + 8*register_elements],t5);
        vst1q_u64((uint64_t*)&helper_CH3[res][col + 10*register_elements],t6);
    }


    /*__m128i xor_1[2];// = _mm_setzero_si128();
    __m128i xor_2[2];// = _mm_setzero_si128();


    xor_1[0] = _mm_setzero_si128();
    xor_1[1] = _mm_setzero_si128();
    xor_2[0] = _mm_setzero_si128();
    xor_2[1] = _mm_setzero_si128();

    for (int i = 0; i < register_elements; i++) {
        xor_1[0] = _mm_xor_si128(reg128_matrix_CH3[j][i], reg128_helper_CH3[rec][i]);
        xor_1[1] = _mm_xor_si128(reg128_matrix_CH3[j][i + register_elements], reg128_helper_CH3[rec][i + register_elements]);
        xor_2[0] = _mm_xor_si128(xor_1[0], reg128_matrix_CH3[j][i + register_elements]);
        xor_2[1] = _mm_xor_si128(xor_1[1], reg128_matrix_CH3[j][i]);

        reg128_helper_CH3[res][i] = _mm_or_si128(xor_2[0], xor_1[1]);
        reg128_helper_CH3[res][i + register_elements] = _mm_or_si128(xor_2[1], xor_1[0]);

        xor_1[0] = _mm_xor_si128(reg128_matrix_CH3[j][i + 2 * register_elements], reg128_helper_CH3[rec][i + 2 * register_elements]);
        xor_1[1] = _mm_xor_si128(reg128_matrix_CH3[j][i + 3 * register_elements], reg128_helper_CH3[rec][i + 3 * register_elements]);

        xor_2[0] = _mm_xor_si128(xor_1[0], reg128_matrix_CH3[j][i + 3 * register_elements]);
        xor_2[1] = _mm_xor_si128(xor_1[1], reg128_matrix_CH3[j][i + 2 * register_elements]);

        reg128_helper_CH3[res][i + 2 * register_elements] = _mm_or_si128(xor_2[0], xor_1[1]);
        reg128_helper_CH3[res][i + 3 * register_elements] = _mm_or_si128(xor_2[1], xor_1[0]);

        xor_1[0] = _mm_xor_si128(reg128_matrix_CH3[j][i + 4 * register_elements], reg128_helper_CH3[rec][i + 4 * register_elements]);
        xor_1[1] = _mm_xor_si128(reg128_matrix_CH3[j][i + 5 * register_elements], reg128_helper_CH3[rec][i + 5 * register_elements]);

        xor_2[0] = _mm_xor_si128(xor_1[0], reg128_matrix_CH3[j][i + 5 * register_elements]);
        xor_2[1] = _mm_xor_si128(xor_1[1], reg128_matrix_CH3[j][i + 4 * register_elements]);
        reg128_helper_CH3[res][i + 4 * register_elements] = _mm_or_si128(xor_2[0], xor_1[1]);
        reg128_helper_CH3[res][i + 5 * register_elements] = _mm_or_si128(xor_2[1], xor_1[0]);
    }*/
}

static inline unsigned long long int weight_GF27_SSEv2(int res) {
    unsigned long long int w64[2]={0,0};
    uint64x2_t help1, help2, element1, element2, element3, result;
    unsigned long long int w = 0;

    for(int i = 0 ; i<2*register_elements; i+=2){
        help1 = vld1q_u64((uint64_t*)&helper_CH3[res][i]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[res][i+2*register_elements]);
        element1 = veorq_u64(help1,help2);

        help1 = vld1q_u64((uint64_t*)&helper_CH3[res][i+4*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[res][i+6*register_elements]);
        element2 = veorq_u64(help1,help2);
        
        help1 = vld1q_u64((uint64_t*)&helper_CH3[res][i+8*register_elements]);
        help2 = vld1q_u64((uint64_t*)&helper_CH3[res][i+10*register_elements]);
        element3 = veorq_u64(help1,help2);

        result = vorrq_u64(element1,element2);
        result = vorrq_u64(result,element3);
        vst1q_u64((uint64_t*)&w64,result);
        w = w + popcount64(w64[0]) + popcount64(w64[1]);
    }
    return w;

    /*__m128i element1 = _mm_setzero_si128(), element2 = _mm_setzero_si128(), element3 = _mm_setzero_si128();
    static union {
        __m128i temp = _mm_setzero_si128();
        unsigned long long  temp64[2];
    };
    unsigned long long int count = 0;

    for (int i = 0; i < register_elements; i++) {
        element1 = _mm_xor_si128(reg128_helper_CH3[res][i], reg128_helper_CH3[res][i + register_elements]);
        element2 = _mm_xor_si128(reg128_helper_CH3[res][i + 2 * register_elements], reg128_helper_CH3[res][i + 3 * register_elements]);
        element3 = _mm_xor_si128(reg128_helper_CH3[res][i + 4 * register_elements], reg128_helper_CH3[res][i + 5 * register_elements]);
        temp = _mm_or_si128(element1, element2);
        temp = _mm_or_si128(temp, element3);
        count = count + popcount(temp64[0]) + popcount(temp64[1]);
    }
    return count;*/
}


void linear_combinations_GF27_SSEv2_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_SSEv2(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_SSEv2(rec);

                if (w < w_searched) {
                    less_than_flag = false;
                }

                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_SSEv2_less_than(rec + 1, j + 1);
                }
            }
        }
    }
}

void linear_combinations_GF27_SSEv2_equal(int rec, int h) {
    if (equal_flag) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_SSEv2(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_SSEv2(rec);

                if (w == w_searched) {
                    equal_flag = false;
                }

                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_SSEv2_equal(rec + 1, j + 1);
                }
            }
        }
    }
}


void linear_combinations_GF27_SSEv2_less_than_count(int rec, int h) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_SSEv2(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_SSEv2(rec);

                if ( w < w_searched) {
                    write_GF27(rec);
                }

                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_SSEv2_less_than_count(rec + 1, j + 1);
                }
            }
        }
}

void linear_combinations_GF27_SSEv2_equal_count(int rec, int h) {
        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_SSEv2(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_SSEv2(rec);

                if ( w == w_searched) {
                    write_GF27(rec);
                }

                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_SSEv2_equal_count(rec + 1, j + 1);
                }
            }
        }
}


void linear_combinations_GF27_SSEv2(int rec, int h) {

        int qf;
        if (h == 1) { qf = 1; }
        else { qf = 26; }
        for (int j = h; j <= K; j++) {
            for (int q1 = 1; q1 <= qf; q1++) {
                if (q1 == 1) {
                    addGF27_SSEv2(j, rec - 1, rec);
                }
                else {
                    int t = TransitionSequence27[q1] - 1;
                    addGF27_SSEv2(t * K + j, rec, rec);
                }

                unsigned long long int w = weight_GF27_SSEv2(rec);
                weights[w]++;
                if (rec < K) {
                    linear_combinations_GF27_SSEv2(rec + 1, j + 1);
                }
            }
        }
}
//---------------------------------Characteristic 3 ----------------------------------------//

//---------------------------------Fields with characteristics 2 and bitwise representation ----------------------------------------//

//sets the registers for composite fields with characteristic 2 and bitwiese representation
void setMatrixGF2_CF(dynamic_mat_short& bits) {
    for (int i = 0; i <= N_FIX*8; i++) {
        weights[i] = 0;
    }
    register_elements = ((N - 1) / 128) + 1;
    int c = ((N - 1) / 64) + 1;
    int bit1 = 0;
    if (c == 1 || c % 2 == 0) {
        bit1 = c;
    }
    else {
        bit1 = 2 * register_elements;
    }
    for (int col = 0; col < (N_GF2); col++) {
        matrix_GF2[0][col] = (0);
        helper_GF2[0][col] = (0);
    }

    for (int row = 1; row <= (M * (K + 1)); row++) {
        for (int col = 0; col < 4*register_elements; col++) {
            matrix_GF2[row][col] = (0);
            helper_GF2[row][col] = (0);
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


// ----------------------- functions for GF4 n<=64 --------------- //
static inline void add_GF4_64(int rec, int i, int res){
    uint64x2_t mat, help, result;
    mat = vld1q_u64((uint64_t*)&matrix_GF2[i][0]);
    help = vld1q_u64((uint64_t*)&helper_GF2[rec][0]);
    result = veorq_u64(mat,help);
    vst1q_u64((uint64_t*)&helper_GF2[res][0],result);
    //reg128_helper_GF2[res][0] = _mm_xor_si128(reg128_helper_GF2[rec][0], reg128_matrix_GF2[i][0]);
}


static inline unsigned long long int weight_GF4_64(int res) {
    unsigned long long  temp = helper_GF2[res][0] | helper_GF2[res][1];
    unsigned long long int weight = popcount64(temp);
    return weight;
}
void linear_combinations_GF4_64_less_than(int rec, int h) {
     if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_64(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_64(rec, t * K + i, rec);
                }
                unsigned long long int weight = weight_GF4_64(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_64_less_than(rec + 1, i + 1);
                }
            }
        }
     }

}


void write_CF2(int res) {
    if (file!=NULL) {
        unsigned long long int one = 1;
        int c = ((N - 1) / 64) + 1;
        int bit1 = 0;
        if (c == 1 || c % 2 == 0) {
            bit1 = c;
        }
        else {
            bit1 = 2 * register_elements;
        }
        for (int el = 0; el < c; el++) {
            if ((el) * 64 > N) { fprintf(file, "\n"); return; }
            for (int shift = 0; shift < 64; shift++) {
                int result = 0;
                if (el * 64 + shift > (N - 1)) { fprintf(file, "\n");  return; }
                for (int m = 0; m < M; m++) {
                    if (helper_GF2[res][el + m *  bit1] & (one << (63 - shift))) {
                        result = result + (1 << m);
                    }
                }
                if (form) {
                    write_multpl(result,  file);
                }
                else {
                    fprintf(file, "%d, ", result);
                }
            }
        }
        fprintf(file, "\n");
    }
}


void linear_combinations_GF4_64_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_64(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_64(rec, t * K + i, rec);
                }
                unsigned long long int weight = weight_GF4_64(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_64_equal(rec + 1, i + 1);
                }
            }
        }
    }

}


void linear_combinations_GF4_64_less_than_count(int rec, int h) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_64(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_64(rec, t * K + i, rec);
                }
                unsigned long long int weight = weight_GF4_64(rec);
                if ( weight < w_searched) {
                    write_CF2(rec);
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_64_less_than_count(rec + 1, i + 1);
                }
            }
        }
}


void linear_combinations_GF4_64_equal_count(int rec, int h) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_64(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_64(rec, t * K + i, rec);
                }
                unsigned long long int weight = weight_GF4_64(rec);
                if ( weight == w_searched) {
                    write_CF2(rec);
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_64_equal_count(rec + 1, i + 1);
                }
            }
        }
}


void linear_combinations_GF4_64(int rec, int h) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_64(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_64(rec, t * K + i, rec);
                }
                unsigned long long int weight = weight_GF4_64(rec);
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_64(rec + 1, i + 1);
                }
            }
        }
}

// ------------------------- functions for GF4 n>64 && n<=128 -------------------------------//
static inline void add_GF4_128(int rec, int i, int res) {
    uint64x2_t mat, help, result;
    mat = vld1q_u64((uint64_t*)&matrix_GF2[i][0]);
    help = vld1q_u64((uint64_t*)&helper_GF2[rec][0]);
    result = veorq_u64(mat,help);
    vst1q_u64((uint64_t*)&helper_GF2[res][0],result);

    mat = vld1q_u64((uint64_t*)&matrix_GF2[i][2]);
    help = vld1q_u64((uint64_t*)&helper_GF2[rec][2]);
    result = veorq_u64(mat,help);
    vst1q_u64((uint64_t*)&helper_GF2[res][2],result);
    //reg128_helper_GF2[res][0] = _mm_xor_si128(reg128_helper_GF2[rec][0], reg128_matrix_GF2[i][0]);
    //reg128_helper_GF2[res][1] = _mm_xor_si128(reg128_helper_GF2[rec][1], reg128_matrix_GF2[i][1]);
}


static inline unsigned long long int weight_GF4_128(int res) {
    uint64x2_t help1, help2, result;

    help2 = vld1q_u64((uint64_t*)&helper_GF2[res][2]);
    help1 = vld1q_u64((uint64_t*)&helper_GF2[res][0]);
    result = vorrq_u64(help1, help2);
    unsigned long long int res64[2] = {0,0};
    vst1q_u64((uint64_t*)&res64, result);

    unsigned long long int weight = popcount64(res64[0]) + popcount64(res64[1]);
    //unsigned long long  int* temp;
    //__m128i temp_reg = _mm_setzero_si128();
    //temp_reg = _mm_or_si128(reg128_helper_GF2[res][0], reg128_helper_GF2[res][1]);
    //temp = (unsigned long long int*) & temp_reg;
    //int weight = popcount(temp[0]) + popcount(temp[1]);
    return weight;
}

void linear_combinations_GF4_128_SSE_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_128(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_128(rec, t * K + i, rec);
                }
                int weight = weight_GF4_128(rec);
                if (weight < w_searched) {
                    less_than_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_128_SSE_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}

void linear_combinations_GF4_128_SSE_equal(int rec, int h) {
    if (equal_flag) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_128(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_128(rec, t * K + i, rec);
                }
                int weight = weight_GF4_128(rec);
                if (weight == w_searched) {
                    equal_flag = false;
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_128_SSE_equal(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_GF4_128_SSE_less_than_count(int rec, int h) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_128(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_128(rec, t * K + i, rec);
                }
                int weight = weight_GF4_128(rec);
                if ( weight < w_searched) {
                    write_CF2(rec);
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_128_SSE_less_than_count(rec + 1, i + 1);
                }
            }
        }
}

void linear_combinations_GF4_128_SSE_equal_count(int rec, int h) {
        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_128(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_128(rec, t * K + i, rec);
                }
                int weight = weight_GF4_128(rec);
                if ( weight == w_searched) {
                    write_CF2(rec);
                }
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_128_SSE_equal_count(rec + 1, i + 1);
                }
            }
        }
}


void linear_combinations_GF4_128_SSE(int rec, int h) {

        int qf = 4;
        if (h == 1) { qf = 2; }
        //else { qf = 4;}
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_GF4_128(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_GF4_128(rec, t * K + i, rec);
                }
                int weight = weight_GF4_128(rec);
                weights[weight]++;
                if (rec < K) {
                    linear_combinations_GF4_128_SSE(rec + 1, i + 1);
                }
            }
        }
}


// ---------------------------------------- functions for fields with characteristic 2 and n>128 ---------------------------------------//
static inline void add_CF2(int rec, int i, int res) {
    uint64x2_t mat, help, result;

    for (int reg = 0; reg < 2*register_elements; reg+=2) {
            mat = vld1q_u64((uint64_t*)&matrix_GF2[i][reg]);
            help = vld1q_u64((uint64_t*)&helper_GF2[rec][reg]);
            result = veorq_u64(mat,help);
            vst1q_u64((uint64_t*)&helper_GF2[res][reg],result);

            mat = vld1q_u64((uint64_t*)&matrix_GF2[i][reg+ 2*register_elements]);
            help = vld1q_u64((uint64_t*)&helper_GF2[rec][reg + 2*register_elements]);
            result = veorq_u64(mat,help);
            vst1q_u64((uint64_t*)&helper_GF2[res][reg + 2*register_elements],result);
         //   reg128_helper_GF2[res][ register_elements + reg] =
          //      _mm_xor_si128(reg128_helper_GF2[rec][register_elements + reg], reg128_matrix_GF2[i][register_elements + reg]);
          
         //   reg128_helper_GF2[res][ reg] =
          //      _mm_xor_si128(reg128_helper_GF2[rec][reg], reg128_matrix_GF2[i][reg]);
        
    }
}



static inline unsigned long long int weight_CF2(int res) {
    uint64x2_t help1, help2, result;
    unsigned long long int res64[2] = {0,0};
    unsigned long long int weight = 0;

    for (int reg = 0; reg < 2*register_elements; reg+=2) {
        help2 = vld1q_u64((uint64_t*)&helper_GF2[res][reg]);
        help1 = vld1q_u64((uint64_t*)&helper_GF2[res][reg + 2*register_elements]);
        result = vorrq_u64(help1, help2);
        vst1q_u64((uint64_t*)&res64, result);
        weight = weight + popcount64(res64[0]) + popcount64(res64[1]);
    }
    return weight;

    //int w = 0;
    //for (int reg = 0; reg < register_elements; reg++) {
     //   temp_reg = _mm_setzero_si128();
     //   for (int m = 0; m < M; m++) {

        //     temp_reg = _mm_or_si128(temp_reg, reg128_helper_GF2[res][m * register_elements + reg]);
        //}
        //unsigned long long* temp = (unsigned long long*) & temp_reg;
        //w = w + popcount(temp[0]) + popcount(temp[1]);
    //}
    //return w;

}



void linear_combinations_CF2_SSE_less_than(int rec, int h) {
    if (less_than_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF2(rec - 1,i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CF2(rec,t * K + i,  rec);
                }
                unsigned long long int w = weight_CF2(rec);
                if (w < w_searched) {
                    less_than_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CF2_SSE_less_than(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_CF2_SSE_equal(int rec, int h) {
    if (equal_flag) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CF2(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF2(rec);
                if (w == w_searched) {
                    equal_flag = false;
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CF2_SSE_equal(rec + 1, i + 1);
                }
            }
        }
    }
}


void linear_combinations_CF2_SSE_less_than_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CF2(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF2(rec);
                if (w < w_searched) {
                    write_CF2(rec);
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CF2_SSE_less_than_count(rec + 1, i + 1);
                }
            }
        }
}


void linear_combinations_CF2_SSE_equal_count(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CF2(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF2(rec);
                if (w == w_searched) {
                    write_CF2(rec);
                }
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CF2_SSE_equal_count(rec + 1, i + 1);
                }
            }
        }
}


void linear_combinations_CF2_SSE(int rec, int h) {
        int qf = Q;
        if (h == 1) { qf = 2; }
        for (int i = h; i <= K; i++) {
            for (int q1 = 1; q1 < qf; q1++) {
                if (q1 == 1) {
                    add_CF2(rec - 1, i, rec);
                }
                else {
                    int t = TransitionSequence64[q1] - 1;
                    add_CF2(rec, t * K + i, rec);
                }
                unsigned long long int w = weight_CF2(rec);
                weights[w]++;
                if (rec < K) {
                    linear_combinations_CF2_SSE(rec + 1, i + 1);
                }
            }
        }
}


//---------------------------------Fields with characteristics 2 and bitwise representation ----------------------------------------//


//------------------------------------  GF2  ----------------------------------------------//

//setting data in registers for n<=64
void set_64_128(dynamic_mat_short &bits) {
    matrix_GF2[0][0] = (0);
    helper_GF2[0][0] = (0);


    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        matrix_GF2[row][0] = (0);
        helper_GF2[row][0] = (0);


        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][0];

    }

    //using coset for calculation of two codewords at the same time
    helper_GF2[0][1] = bits.a[K - 1][0];


}

// setting data in registers for n>64 && n<=128
void set_128_128(dynamic_mat_short &bits) {

  //  cout << "Set registers cosets (128_128)" << endl;
    matrix_GF2[0][0] = (0);
    helper_GF2[0][0] = (0);

    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        matrix_GF2[row][0] = (0);
        helper_GF2[row][0] = (0);

        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][1];

    }

}

// setting data in registers for n>128 && n<256
void set_256_128(dynamic_mat_short &bits) {

    matrix_GF2[0][0] = (0);
    helper_GF2[0][0] = (0);

    matrix_GF2[0][1] = (0);
    helper_GF2[0][1] = (0);

    for (int i = 0; i < N; i++) {
        weights[i] = 0;
    }
    for (int row = 1; row <= K; row++) {
        matrix_GF2[row][0] = (0);
        helper_GF2[row][0] = (0);

        matrix_GF2[row][1] = (0);
        helper_GF2[row][1] = (0);

        matrix_GF2[row][0] = bits.a[row - 1][0];
        matrix_GF2[row][1] = bits.a[row - 1][1];
        matrix_GF2[row][2] = bits.a[row - 1][2];
        matrix_GF2[row][3] = bits.a[row - 1][3];

    }

}

// setting data into registers for n>256
void set_512_128(dynamic_mat_short& bits) {
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }
    for (int col = 0; col < 2*register_elements; col++) {
        matrix_GF2[0][col] = (0);
        helper_GF2[0][col] = (0);
    }
    for (int row = 1; row <= K; row++) {
        for (int col = 0; col < 2*register_elements; col++) {
            matrix_GF2[row][col] = (0);
            helper_GF2[row][col] = (0);
        }
        for (int el = 0; el < N_GF2; el++) {
            if ((el) * 64 > N) {
                break;
            }
            matrix_GF2[row][el] = bits.a[row - 1][el];
        }
    }
}

// addition for GF2 n < 128
static inline void add_GF2_128_128(int rec, int i, int res) {
    uint64x2_t v1 = vld1q_u64 ((uint64_t*)&helper_GF2[rec][0]);
    uint64x2_t v2 = vld1q_u64 ((uint64_t*)&matrix_GF2[i][0]);
    uint64x2_t r = veorq_u64(v1,v2);
    vst1q_u64((uint64_t*)&helper_GF2[res][0],r);
    //not working for 64<n<128
    //reg128_helper_GF2[res][0] = _mm_xor_si128(reg128_helper_GF2[rec][0], reg128_matrix_GF2[i][0]);
}

//writing codeword into a file for GF2
 void write_GF2(int res) {
    if (file!=NULL) {
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

// writing codewords into file for n<=64 (using cosets)
void write_GF2_coset(int res, int el) {
    if (file!=NULL) {
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

//------------------------------- functions for calculations for n<=64 --------------------------------//
void linear_combinations_64_128_less_than(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j < K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount64(helper_GF2[rec][0]);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;

            w = popcount64(helper_GF2[rec][1]);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;
            if (rec < K - 1) {
                linear_combinations_64_128_less_than(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_64_128_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j < K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount64(helper_GF2[rec][0]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            w = popcount64(helper_GF2[rec][1]);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;
            if (rec < K - 1) {
                linear_combinations_64_128_equal(rec + 1, j + 1);
            }
        }
    }
}


void linear_combinations_64_128_less_than_count(int rec, int h) {
        for (int j = h; j < K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount64(helper_GF2[rec][0]);
            if (w < w_searched) {
                write_GF2_coset(rec,0);
            }
            weights[w]++;
            w = 0;
            w = popcount64(helper_GF2[rec][1]);
            if (  w < w_searched) {
                write_GF2_coset(rec,1);
            }
            weights[w]++;
            if (rec < K - 1) {
                linear_combinations_64_128_less_than_count(rec + 1, j + 1);
            }
        }
}

void linear_combinations_64_128_equal_count(int rec, int h) {
        for (int j = h; j < K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount64(helper_GF2[rec][0]);
            if (w == w_searched) {
                write_GF2_coset(rec,0);
            }
            weights[w]++;

            w = popcount64(helper_GF2[rec][1]);
            if ( w == w_searched) {
                write_GF2_coset(rec,1);
            }
            weights[w]++;
            if (rec < K - 1) {
                linear_combinations_64_128_equal_count(rec + 1, j + 1);
            }
        }
}



void linear_combinations_64_128(int rec, int h) {
        for (int j = h; j < K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned long long int w = 0;
            w = popcount64(helper_GF2[rec][0]);
            weights[w]++;
            //cout<<"4<<w<<endl;

            w = popcount64(helper_GF2[rec][1]);
            weights[w]++;
            //cout<<"w="<<w<<endl;
            if (rec < K - 1) {
                linear_combinations_64_128(rec + 1, j + 1);
            }
        }
}


// function for calculation the weight for GF2 n>64 && n<=128
static inline unsigned long long int weight_GF2_128_128(int res) {
    unsigned long long int w = popcount64(helper_GF2[res][0]) + popcount64(helper_GF2[res][1]);
    //not working for 64<n<128
    return w;
}

// ---------------------------- functions for GF2 n>64 && n<=128 ------------------------------------//
void linear_combinations_128_128_less_than(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned int w = weight_GF2_128_128(rec);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_128_128_less_than(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_128_128_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned int w = weight_GF2_128_128(rec);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_128_128_equal(rec + 1, j + 1);
            }
        }
    }
}

void linear_combinations_128_128_less_than_count(int rec, int h) {
        for (int j = h; j <= K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned int w = weight_GF2_128_128(rec);
            if ( w < w_searched) {
                write_GF2(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_128_128_less_than_count(rec + 1, j + 1);
            }
        }
}

void linear_combinations_128_128_equal_count(int rec, int h) {
        for (int j = h; j <= K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned int w = weight_GF2_128_128(rec);
            if ( w == w_searched) {
                write_GF2(rec);
            }
            weights[w]++;

            if (rec < K) {
                linear_combinations_128_128_equal_count(rec + 1, j + 1);
            }
        }
}

void linear_combinations_128_128(int rec, int h) {
        for (int j = h; j <= K; j++) {
            add_GF2_128_128(rec - 1, j, rec);
            unsigned int w = weight_GF2_128_128(rec);
            weights[w]++;
            //cout<<"w="<<w<<endl;

            if (rec < K) {
                linear_combinations_128_128(rec + 1, j + 1);
            }
        }

}

/*
void linear_combinations_256_128(int rec, int h) {
   // if (less_than_flag) {
        for (int j = h; j <= K; j++) {
            reg128_helper_GF2[rec][0] = _mm_xor_si128(reg128_helper_GF2[rec - 1][0], reg128_matrix_GF2[j][0]);
            reg128_helper_GF2[rec][1] = _mm_xor_si128(reg128_helper_GF2[rec - 1][1], reg128_matrix_GF2[j][1]);
            unsigned int w = 0;
            w = popcount(helper_GF2[rec][0]) + popcount(helper_GF2[rec][1]) +
                popcount(helper_GF2[rec][2]) + popcount(helper_GF2[rec][3]);
            //if (w < w_searched) {
           //     less_than_flag = false;
           //}
            weights[w]++;
            if (rec < K) {
                linear_combinations_256_128(rec + 1, j + 1);
            }
        }
    //}

}
*/

// ------------------------------- functions for GF2  n>128 -------------------------//
static inline void add_GF2_128(int rec, int i, int res) {
    for (int el = 0; el < register_elements; el++) {
        uint64x2_t v1 = vld1q_u64 ((uint64_t*)&helper_GF2[rec][2*el]);
        uint64x2_t v2 = vld1q_u64 ((uint64_t*)&matrix_GF2[i][2*el]);
        uint64x2_t r = veorq_u64(v1,v2);
        vst1q_u64((uint64_t*)&helper_GF2[res][2*el],r);
      //  reg128_helper_GF2[res][el] = _mm_xor_si128(reg128_helper_GF2[rec][el], reg128_matrix_GF2[i][el]);
    }
}


static inline unsigned long long int weight_GF2_128(int res) {
    unsigned long long int w = 0;
    for (int el = 0; el < register_elements; el++) {
        w = w + popcount64(helper_GF2[res][2 * el]) + popcount64(helper_GF2[res][2 * el + 1]);

    }
    return w;
}


void linear_combinations_512_128_less_than(int rec, int h) {
    if (less_than_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_128(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_128(rec);
            if (w < w_searched) {
                less_than_flag = false;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_128_less_than(rec + 1, j + 1);
            }
        }
    }

}

void linear_combinations_512_128_equal(int rec, int h) {
    if (equal_flag) {
        for (int j = h; j <= K; j++) {
            add_GF2_128(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_128(rec);
            if (w == w_searched) {
                equal_flag = false;
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_128_equal(rec + 1, j + 1);
            }
        }
    }

}


void linear_combinations_512_128_less_than_count(int rec, int h) {
        for (int j = h; j <= K; j++) {
            add_GF2_128(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_128(rec);
            if (  w < w_searched) {
                write_GF2(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_128_less_than_count(rec + 1, j + 1);
            }
        }
}

void linear_combinations_512_128_equal_count(int rec, int h) {
        for (int j = h; j <= K; j++) {
            add_GF2_128(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_128(rec);
            if ( w == w_searched) {
                write_GF2(rec);
            }
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_128_equal_count(rec + 1, j + 1);
            }
        }
}


void linear_combinations_512_128(int rec, int h) {
        for (int j = h; j <= K; j++) {
            add_GF2_128(rec - 1, j, rec);
            unsigned long long int w = weight_GF2_128(rec);
            weights[w]++;
            if (rec < K) {
                linear_combinations_512_128(rec + 1, j + 1);
            }
        }
}


//------------------------------------  GF2  ----------------------------------------------//


// --------------------------------------- main functions for calculations ---------------------------------//
//
// each function is called for specific field with different characteristic and representation of the elements
// in the function the correct implementation of the calculation of the linear combination is chosen based on q
// only nonproportional codewords are genereted in the calculations
// each function calles the appropriate function for setting the registers
// at the end the full weight spectrum is calculated
//
//-------------------weight calculation functions ------------------------------//
void calculateWeightCH3_neon(dynamic_mat_short &bits, int n, int k, int m) {

    //popcnt_detect();
    K = k;
    N = n;
    M = m;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    register_elements = (((n - 1) / 128) + 1);
   // cout << "num of registers = " << register_elements << endl;
   // cout << "N = " << N << "  K = " << K << endl;
   // clock_t begin, end;
   // begin = clock();
    if (m == 1) {
        Q = 3;
        setMatrixGF3(bits);
        if (n <= 64) {

            linear_comb_recGF3_64(1, 1);
            // linear_comb_recGF3_SSE(1, 1);

        }
        else {
            linear_comb_recGF3_SSE(1, 1);
        }


    }
    else if (m == 2) {
        Q = 9;
       // cout << "calc weight(GF9)" << endl;
        setMatrixGF9v2(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_SSEv2(1, 1);

        }
        else {
            linear_combinations_GF9_SSEv2(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
     //   cout << "calc weight(GF27)" << endl;
        setMatrixGF27v2(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_SSEv2(1, 1);

        }
        else {
            linear_combinations_GF27_SSEv2(1, 1);
        }

    }
  //  end = clock();
   // double t = (end - begin) / (double(CLOCKS_PER_SEC));
   // ofstream out;
  //  out.open("RES.txt", ios::app);
 //   out << "Time (128-bit register): " << t << "s\n";
 //   out.close();
    //printf("Time (weight first version): %.8fs\n\n", t);
    //printWeights(weights, N, Q);
    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i]* (Q - 1);
    }

}


void calculateWeightBytes_neon(dmat_type &bits, int n, int k, int m, int q) {
   
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    less_than_flag = true;
    register_elements = ((n - 1) / 128) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes(bits);
            linear_combinations_CH2(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersCF(bits);
            linear_combinations_CF_49(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersCF(bits);
            linear_combinations_CF_25(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes(bits);
        linear_combinations_Bytes(1, 1);
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
}


void calculateWeightCH2_neon(dynamic_mat_short& bits, int n, int k, int m) {
    
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    //less_than_flag = true;
    register_elements = ((n - 1) / 128) + 1;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    setMatrixGF2_CF(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_SSE(1, 1);
        }
        else {
            linear_combinations_CF2_SSE(1, 1);
        }
        break;
   /* case 8:
        if (n <= 64) {
            linear_combinations_GF8_64(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF8_128_SSE(1, 1);
        }
        else {
            linear_combinations_CF2_SSE(1, 1);
        }
        break;
    case 16:
        if (n <= 64) {
            linear_combinations_GF16_64(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF16_128_SSE(1, 1);
        }
        else {
            linear_combinations_CF2_SSE(1, 1);
        }
        break;
    case 32:
        if (n <= 64) {
            linear_combinations_GF32_64(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF32_128_SSE(1, 1);
        }
        else {
            linear_combinations_CF2_SSE(1, 1);
        }
        break;
    case 64:
        if (n <= 64) {
            linear_combinations_GF64_64(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF64_128_SSE(1, 1);
        }
        else {
            linear_combinations_CF2_SSE(1, 1);
        }
        break;
        */
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
}



void calculateWeightGF2_neon(dynamic_mat_short& bits, int n, int k) {

   // popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    less_than_flag = true;

    register_elements = ((n - 1) / 128) + 1;
   // cout << "calc weight" << endl;
   // cout << "N = " << N << "  K = " << K << endl;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    clock_t begin, end;
    begin = clock();

    if (n <= 64) {
        set_64_128(bits);
        unsigned int w = popcount64(helper_GF2[0][1]);
        weights[w]++;
        linear_combinations_64_128(1, 1);
    }
    //else if (n <= 128) {
     //   set_128_128(bits);
     //   linear_combinations_128_128(1, 1);
    //}
    else {
        set_512_128(bits);
        linear_combinations_512_128(1, 1);

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));
    //printf("Time (weight): %.8fs\n\n", t);
   // ofstream out;
   // out.open("RES.txt", ios::app);
  //  out << "Time (128-bit register): " << t << "s\n";
 //   out.close();
  //  printWeights(weights, N, Q);

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }

  //  cout << endl;
}

//------------------------------ Count the number of code words with w < w_fixed -----------------------------//

unsigned long long int calculateNumberOfWordsCH3_neon_less_than(dynamic_mat_short &bits, int n, int k, int m,  int d, bool multiplicativeForm) {

    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    form = multiplicativeForm;
    Q = 1;
    for (int i = 0; i < m; i++) {
        Q = Q * 3;
    }
    file = fopen("Result_codewords.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight < %llu:\n", w_searched);
    }
    else {
        cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 128) + 1);
    // cout << "N = " << N << "  K = " << K << endl;
    clock_t begin, end;
    begin = clock();
    if (m == 1) {
      //  Q = 3;
        setMatrixGF3(bits);
        if (n <= 64) {

            linear_comb_recGF3_64_less_than_count(1, 1);
            // linear_comb_recGF3_SSE(1, 1);

        }
        else {
            linear_comb_recGF3_SSE_less_than_count(1, 1);
        }


    }
    else if (m == 2) {
       // Q = 9;
        // cout << "calc weight(GF9)" << endl;
        setMatrixGF9v2(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_SSEv2_less_than_count(1, 1);

        }
        else {
            linear_combinations_GF9_SSEv2_less_than_count(1, 1);
        }
    }
    else if (m == 3) {
      //  Q = 27;
        //   cout << "calc weight(GF27)" << endl;
        setMatrixGF27v2(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_SSEv2_less_than_count(1, 1);

        }
        else {
            linear_combinations_GF27_SSEv2_less_than_count(1, 1);
        }

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));

    unsigned long long int ct = 0;
    unsigned long long int  i = 0;
    while (i < w_searched) {
        ct = ct + weights[i];
        i++;
    }
    fprintf(file, "\n\n");
    fclose(file);

    return ct;

}


unsigned long long int  calculateNumberOfWordsBytes_neon_less_than(dmat_type &bits, int n, int k, int m, int q, int d, bool multiplicativeForm) {
    //popcnt_detect();
    K = k;
    N = n;
    M = m;

    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords.txt", "a");



    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    clock_t begin, end;

    double t = 0;
    begin = clock();
    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight < %llu:\n", w_searched);
            }
            else {
                cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
            }
            setRegistersBytes(bits);
            linear_combinations_CH2_less_than_count(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight < %llu:\n", w_searched);
            }
            else {
                cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
            }
            setRegistersCF(bits);
            linear_combinations_CF_49_less_than_count(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight < %llu:\n", w_searched);
            }
            else {
                cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
            }
            setRegistersCF(bits);
            linear_combinations_CF_25_less_than_count(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        if (file != NULL) {
            fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
            fprintf(file, "Searching for words with weight < %llu:\n", w_searched);
        }
        else {
            cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
        }
        setRegistersBytes(bits);
        linear_combinations_Bytes_less_than_count(1, 1);
    }

    end = clock();
    t = (end - begin) / (double(CLOCKS_PER_SEC));
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


unsigned long long int calculateNumberOfWordsCH2_neon_less_than(dynamic_mat_short &bits, int n, int k, int m, int d, bool multiplicativeForm) {
   // popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight < %llu:\n", w_searched);
    }
    else {
        cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
    }


    // cout << "calc weight(CF 2) coef" << endl;
    // cout << "N = " << N << "  K = " << K << endl;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }



    setMatrixGF2_CF(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_less_than_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_SSE_less_than_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than_count(1, 1);
        }
        break;
    /*case 8:
        if (n <= 64) {
            linear_combinations_GF8_64_less_than_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF8_128_SSE_less_than_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than_count(1, 1);
        }
        break;
    case 16:
        if (n <= 64) {
            linear_combinations_GF16_64_less_than_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF16_128_SSE_less_than_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than_count(1, 1);
        }
        break;
    case 32:
        if (n <= 64) {
            linear_combinations_GF32_64_less_than_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF32_128_SSE_less_than_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than_count(1, 1);
        }
        break;
    case 64:
        if (n <= 64) {
            linear_combinations_GF64_64_less_than_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF64_128_SSE_less_than_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than_count(1, 1);
        }
        break;*/
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



unsigned long long int calculateNumberOfWordsGF2_neon_less_than(dynamic_mat_short &bits, int n, int k, int d, bool multiplicativeForm) {
    //popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;

    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight < %llu:\n", w_searched);
    }
    else {
        cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
    }

    register_elements = ((n - 1) / 128) + 1;
    // cout << "calc weight" << endl;
    // cout << "N = " << N << "  K = " << K << endl;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    clock_t begin, end;
    begin = clock();

    if (n <= 64) {
        set_64_128(bits);
        unsigned int w = popcount64(helper_GF2[0][1]);
        weights[w]++;
        if (w < w_searched) {
            write_GF2_coset(0,1);
        }
        linear_combinations_64_128_less_than_count(1, 1);
    }
    //else if (n <= 128) {
     //   set_128_128(bits);
     //   linear_combinations_128_128_less_than_count(1, 1);
    //}
    else {
        set_512_128(bits);
        linear_combinations_512_128_less_than_count(1, 1);

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));
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


//------------------------------ Count the number of code words with w = w_fixed -----------------------------//

unsigned long long int calculateNumberOfWordsCH3_neon_equal(dynamic_mat_short &bits, int n, int k, int m, int d, bool multiplicativeForm) {
   // popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    form = multiplicativeForm;

    Q = 1;
    for (int i = 0; i < m; i++) {
        Q = Q * 3;
    }

    file = fopen("Result_codewords.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight = %llu:\n", w_searched);
    }
    else {
        cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
    }


    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 128) + 1);
    // cout << "N = " << N << "  K = " << K << endl;
    clock_t begin, end;
    begin = clock();
    if (m == 1) {
        Q = 3;
        setMatrixGF3(bits);
        if (n <= 64) {

            linear_comb_recGF3_64_equal_count(1, 1);
            // linear_comb_recGF3_SSE(1, 1);

        }
        else {
            linear_comb_recGF3_SSE_equal_count(1, 1);
        }

    }
    else if (m == 2) {
        Q = 9;
        // cout << "calc weight(GF9)" << endl;
        setMatrixGF9v2(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_SSEv2_equal_count(1, 1);

        }
        else {
            linear_combinations_GF9_SSEv2_equal_count(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
        //   cout << "calc weight(GF27)" << endl;
        setMatrixGF27v2(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_SSEv2_equal_count(1, 1);

        }
        else {
            linear_combinations_GF27_SSEv2_equal_count(1, 1);
        }

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));

    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);

    return ct;

}


unsigned long long int  calculateNumberOfWordsBytes_neon_equal(dmat_type &bits, int n, int k, int m, int q, int d, bool multiplicativeForm) {
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords.txt", "a");


    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    clock_t begin, end;

    double t = 0;
    begin = clock();
    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            if (file!=NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight =  %llu:\n", w_searched);
            }
            else {
                cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
            }
            setRegistersBytes(bits);
            linear_combinations_CH2_equal_count(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight =  %llu:\n", w_searched);
            }
            else {
                cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
            }
            setRegistersCF(bits);
            linear_combinations_CF_49_equal_count(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            if (file != NULL) {
                fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
                fprintf(file, "Searching for words with weight =  %llu:\n", w_searched);
            }
            else {
                cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
            }
            setRegistersCF(bits);
            linear_combinations_CF_25_equal_count(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        if (file != NULL) {
            fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
            fprintf(file, "Searching for words with weight =  %llu:\n", w_searched);
        }
        else {
            cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
        }
        setRegistersBytes(bits);
        linear_combinations_Bytes_euqal_count(1, 1);
    }

    end = clock();
    t = (end - begin) / (double(CLOCKS_PER_SEC));
    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}


unsigned long long int calculateNumberOfWordsCH2_neon_equal(dynamic_mat_short &bits, int n, int k, int m, int d, bool multiplicativeForm) {
   // popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    form = multiplicativeForm;

    file = fopen("Result_codewords.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight = %llu:\n", w_searched);
    }
    else {
        cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    clock_t begin, end;
    double t = 0;
    begin = clock();
    setMatrixGF2_CF(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_equal_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_SSE_equal_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal_count(1, 1);
        }
        break;
    /*case 8:
        if (n <= 64) {
            linear_combinations_GF8_64_equal_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF8_128_SSE_equal_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal_count(1, 1);
        }
        break;
    case 16:
        if (n <= 64) {
            linear_combinations_GF16_64_equal_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF16_128_SSE_equal_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal_count(1, 1);
        }
        break;
    case 32:
        if (n <= 64) {
            linear_combinations_GF32_64_equal_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF32_128_SSE_equal_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal_count(1, 1);
        }
        break;
    case 64:
        if (n <= 64) {
            linear_combinations_GF64_64_equal_count(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF64_128_SSE_equal_count(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal_count(1, 1);
        }
        break;*/
    }

    end = clock();
    t = (end - begin) / (double(CLOCKS_PER_SEC));
    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}



unsigned long long int calculateNumberOfWordsGF2_neon_equal(dynamic_mat_short& bits, int n, int k, int d, bool multiplicativeForm) {
    //popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    form = multiplicativeForm;


    file = fopen("Result_codewords.txt", "a");
    if (file != NULL) {
        fprintf(file, "n = %d, k = %d, q = %d\n", N, K, Q);
        fprintf(file, "Searching for words with weight = %llu:\n", w_searched);
    }
    else {
        cout << "Cannot open file Result_codewords.txt\n The codewords won't be writen!\n";
    }
    register_elements = ((n - 1) / 128) + 1;
    // cout << "calc weight" << endl;
    // cout << "N = " << N << "  K = " << K << endl;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    clock_t begin, end;
    begin = clock();

    if (n <= 64) {
        set_64_128(bits);
        unsigned int w = popcount64(helper_GF2[0][1]);
        weights[w]++;
        if (w == w_searched) {
            write_GF2_coset(0,1);
        }
        linear_combinations_64_128_equal_count(1, 1);
    }
   // else if (n <= 128) {
    //    set_128_128(bits);
     //   linear_combinations_128_128_equal_count(1, 1);
   // }
    else {
        set_512_128(bits);
        linear_combinations_512_128_equal_count(1, 1);

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));
    unsigned long long int ct = weights[w_searched];
    fprintf(file, "\n\n");
    fclose(file);
    return ct;
}


// ---------------------- searching for weight less than fixed w ------------------//

bool calculateWeightCH3_neon_less_than(dynamic_mat_short &bits, int n, int k, int m,  int d) {
    
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    less_than_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 128) + 1);
    // cout << "N = " << N << "  K = " << K << endl;
    clock_t begin, end;
    begin = clock();
    if (m == 1) {
        Q = 3;
        setMatrixGF3(bits);
        if (n <= 64) {

            linear_comb_recGF3_64_less_than(1, 1);
            // linear_comb_recGF3_SSE(1, 1);

        }
        else {
            linear_comb_recGF3_SSE_less_than(1, 1);
        }


    }
    else if (m == 2) {
        Q = 9;
        // cout << "calc weight(GF9)" << endl;
        setMatrixGF9v2(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_SSEv2_less_than(1, 1);

        }
        else {
            linear_combinations_GF9_SSEv2_less_than(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
        //   cout << "calc weight(GF27)" << endl;
        setMatrixGF27v2(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_SSEv2_less_than(1, 1);

        }
        else {
            linear_combinations_GF27_SSEv2_less_than(1, 1);
        }

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));
    // ofstream out;
   //  out.open("RES.txt", ios::app);
  //   out << "Time (128-bit register): " << t << "s\n";
  //   out.close();
     //printf("Time (weight first version): %.8fs\n\n", t);
     //printWeights(weights, N, Q);
    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}


bool calculateWeightBytes_neon_less_than(dmat_type &bits, int n, int k, int m, int q,  int d) {
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    less_than_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    clock_t begin, end;

    double t = 0;
    begin = clock();
    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes(bits);
            linear_combinations_CH2_less_than(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersCF(bits);
            linear_combinations_CF_49_less_than(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersCF(bits);
            linear_combinations_CF_25_less_than(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes(bits);
        linear_combinations_Bytes_less_than(1, 1);
    }

    end = clock();
    t = (end - begin) / (double(CLOCKS_PER_SEC));
    //ofstream out;
    // out.open("RES.txt", ios::app);
   //  out << "Time (128-bit register): " << t << "s\n";
   //  out.close();
     //printf("Time: (bytes) %.8fs\n\n", t);
    // printWeights(weights, N, Q);
    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}


bool calculateWeightCH2_neon_less_than(dynamic_mat_short &bits, int n, int k, int m, int d) {
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    less_than_flag = true;

    // cout << "calc weight(CF 2) coef" << endl;
    // cout << "N = " << N << "  K = " << K << endl;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    clock_t begin, end;
    double t = 0;
    begin = clock();
    setMatrixGF2_CF(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_less_than(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_SSE_less_than(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than(1, 1);
        }
        break;
    /*case 8:
        if (n <= 64) {
            linear_combinations_GF8_64_less_than(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF8_128_SSE_less_than(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than(1, 1);
        }
        break;
    case 16:
        if (n <= 64) {
            linear_combinations_GF16_64_less_than(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF16_128_SSE_less_than(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than(1, 1);
        }
        break;
    case 32:
        if (n <= 64) {
            linear_combinations_GF32_64_less_than(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF32_128_SSE_less_than(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than(1, 1);
        }
        break;
    case 64:
        if (n <= 64) {
            linear_combinations_GF64_64_less_than(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF64_128_SSE_less_than(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_less_than(1, 1);
        }
        break;*/
    }

    end = clock();
    t = (end - begin) / (double(CLOCKS_PER_SEC));
    // ofstream out;
   //  out.open("RES.txt", ios::app);
   //  out << "Time (128-bit register): " << t << "s\n";
   //  out.close();
     //printf("Time (weight SSE, GF2 CF): %.8fs\n\n", t);
  //   printWeights(weights,N, Q);
    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}



bool calculateWeightGF2_neon_less_than(dynamic_mat_short& bits, int n, int k, int d) {
    //popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    less_than_flag = true;

    register_elements = ((n - 1) / 128) + 1;
    // cout << "calc weight" << endl;
    // cout << "N = " << N << "  K = " << K << endl;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }



    if (n <= 64) {
        set_64_128(bits);
        unsigned int w = popcount64(helper_GF2[0][1]);
        if (w < w_searched) { return true; }
        weights[w]++;

        linear_combinations_64_128_less_than(1, 1);
    }
   // else if (n <= 128) {
   //     set_128_128(bits);
   //     linear_combinations_128_128_less_than(1, 1);
   // }
    else {
        set_512_128(bits);
        linear_combinations_512_128_less_than(1, 1);

    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !less_than_flag;
}



// ---------------------- searcher for weight equal to fied w ------------------//

bool calculateWeightCH3_neon_equal(dynamic_mat_short &bits, int n, int k, int m, int d) {
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    equal_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    register_elements = (((N - 1) / 128) + 1);
    // cout << "N = " << N << "  K = " << K << endl;
    clock_t begin, end;
    begin = clock();
    if (m == 1) {
        Q = 3;
        setMatrixGF3(bits);
        if (n <= 64) {

            linear_comb_recGF3_64_equal(1, 1);
            // linear_comb_recGF3_SSE(1, 1);

        }
        else {
            linear_comb_recGF3_SSE_equal(1, 1);
        }


    }
    else if (m == 2) {
        Q = 9;
        // cout << "calc weight(GF9)" << endl;
        setMatrixGF9v2(bits);
        if (n <= 64) {
            linear_combinations_GF9_64_SSEv2_equal(1, 1);

        }
        else {
            linear_combinations_GF9_SSEv2_equal(1, 1);
        }
    }
    else if (m == 3) {
        Q = 27;
        //   cout << "calc weight(GF27)" << endl;
        setMatrixGF27v2(bits);

        if (n <= 64) {
            linear_combinations_GF27_64_SSEv2_equal(1, 1);

        }
        else {
            linear_combinations_GF27_SSEv2_equal(1, 1);
        }

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));
    // ofstream out;
   //  out.open("RES.txt", ios::app);
  //   out << "Time (128-bit register): " << t << "s\n";
  //   out.close();
     //printf("Time (weight first version): %.8fs\n\n", t);
     //printWeights(weights, N, Q);
    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}


bool calculateWeightBytes_neon_equal(dmat_type &bits, int n, int k, int m, int q, int d) {
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    w_searched = d;
    equal_flag = true;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }

    clock_t begin, end;

    double t = 0;
    begin = clock();
    if (m > 1) {
        Characteristic = q;
        if (q == 2) {
            Q = 1 << m;
            setRegistersBytes(bits);
            linear_combinations_CH2_equal(1, 1);
        }
        else if (q == 7) {
            Q = q * q;
            setRegistersCF(bits);
            linear_combinations_CF_49_equal(1, 1);
        }
        else if (q == 5) {
            Q = q * q;
            setRegistersCF(bits);
            linear_combinations_CF_25_equal(1, 1);
        }
    }
    else {
        Characteristic = q;
        Q = q;
        setRegistersBytes(bits);
        linear_combinations_Bytes_euqal(1, 1);
    }

    end = clock();
    t = (end - begin) / (double(CLOCKS_PER_SEC));
    //ofstream out;
    // out.open("RES.txt", ios::app);
   //  out << "Time (128-bit register): " << t << "s\n";
   //  out.close();
     //printf("Time: (bytes) %.8fs\n\n", t);
    // printWeights(weights, N, Q);
    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}


bool calculateWeightCH2_neon_equal(dynamic_mat_short &bits, int n, int k, int m, int d) {
    //popcnt_detect();
    K = k;
    N = n;
    M = m;
    Q = 1 << m;
    w_searched = d;
    equal_flag = true;

    // cout << "calc weight(CF 2) coef" << endl;
    // cout << "N = " << N << "  K = " << K << endl;

    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    setMatrixGF2_CF(bits);
    switch (Q) {
    case 4:
        if (n <= 64) {
            linear_combinations_GF4_64_equal(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF4_128_SSE_equal(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal(1, 1);
        }
        break;
    /*case 8:
        if (n <= 64) {
            linear_combinations_GF8_64_equal(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF8_128_SSE_equal(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal(1, 1);
        }
        break;
    case 16:
        if (n <= 64) {
            linear_combinations_GF16_64_equal(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF16_128_SSE_equal(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal(1, 1);
        }
        break;
    case 32:
        if (n <= 64) {
            linear_combinations_GF32_64_equal(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF32_128_SSE_equal(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal(1, 1);
        }
        break;
    case 64:
        if (n <= 64) {
            linear_combinations_GF64_64_equal(1, 1);
        }
        else if (n <= 128) {
            linear_combinations_GF64_128_SSE_equal(1, 1);
        }
        else {
            linear_combinations_CF2_SSE_equal(1, 1);
        }
        break;*/
    }

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
}



bool calculateWeightGF2_neon_equal(dynamic_mat_short &bits, int n, int k, int d) {
    //popcnt_detect();
    K = k;
    N = n;
    Q = 2;
    M = 1;
    w_searched = d;
    equal_flag = true;

    register_elements = ((n - 1) / 128) + 1;
    // cout << "calc weight" << endl;
    // cout << "N = " << N << "  K = " << K << endl;
    for (int i = 0; i <= N; i++) {
        weights[i] = 0;
    }


    clock_t begin, end;
    begin = clock();

    if (n <= 64) {
        set_64_128(bits);
        unsigned int w = popcount64(helper_GF2[0][1]);
        if (w == w_searched) return true;
        weights[w]++;
        linear_combinations_64_128_equal(1, 1);
    }
    //else if (n <= 128) {
    //    set_128_128(bits);
    //    linear_combinations_128_128_equal(1, 1);
   // }
    else {
        set_512_128(bits);
        linear_combinations_512_128_equal(1, 1);

    }
    end = clock();
    double t = (end - begin) / (double(CLOCKS_PER_SEC));
    //printf("Time (weight): %.8fs\n\n", t);
   // ofstream out;
   // out.open("RES.txt", ios::app);
  //  out << "Time (128-bit register): " << t << "s\n";
 //   out.close();
  //  printWeights(weights, N, Q);

    for (int i = 0; i <= N; i++) {
        weights[i] = weights[i] * (Q - 1);
    }
    return !equal_flag;
    //  cout << endl;
}
