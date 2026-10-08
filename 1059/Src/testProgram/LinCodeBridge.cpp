// LinCodeBridge.cpp
//
// Minimal, script-friendly CLI wrapper around LinCodeWeightInv's in-memory
// API (Algorithm 1059, Pashinska-Gadzheva & Bouyukliev, ACM TOMS 2025),
// written for lincodeweightinv_analysis.ipynb in the PEP-LEP repo.
//
// WHY A CUSTOM BRIDGE INSTEAD OF LinCodeWeightInv's OWN FILE FORMAT:
// LinCodeWeightInv's own text-file generator-matrix reader (readMatrix in
// ReadWrite.cpp) reads AT MOST 2 decimal digit characters per field
// element whenever q > 9 (see the "numberString[2]" / charStriingToInt(_,
// 2) logic there). That means it cannot correctly parse a field element
// >= 100. This project's codes use q = 127 (elements 0..126), and a
// uniformly random GF(127) matrix has roughly a 21% chance per entry of
// being >= 100 - so LinCodeWeightInv's own file format would silently
// corrupt the matrix for essentially every real instance here.
//
// This bridge sidesteps that bug completely: it never calls
// LinCodeWeightInv's file-based entry points at all. Instead it does its
// own trivial, digit-width-unlimited parsing of a much simpler input
// format (below), builds the int** generatorMatrix array directly, and
// calls the in-memory API (calculateWeightDistribution /
// calculate_number_of_words_less_than_fixed_w) with that array.
//
// Usage:
//   lincode_bridge spectrum  <matrix_file>
//   lincode_bridge less_than <matrix_file> <W>
//   lincode_bridge equal     <matrix_file> <W>
//
// <matrix_file> format (this bridge's own format, NOT LinCodeWeightInv's
// "? k n q num" format):
//   line 1: n k q
//   then k*n whitespace-separated decimal integers in [0, q-1] (the
//   generator matrix, K rows x N columns, in row-major order - line
//   breaks beyond whitespace don't matter, fscanf just consumes k*n ints)
//
// Output (to stdout):
//   mode "spectrum":
//     ===WEIGHTS===
//     <weight>,<count>        (one line per weight i = 0..n, in order;
//                               count is 0 for weights with no codewords)
//     ===END===
//   mode "less_than":
//     COUNT:<unsigned long long>   (number of codewords of weight < W,
//                                    counting proportional multiples -
//                                    i.e. this is calculate_number_of_
//                                    words_less_than_fixed_w's return
//                                    value, unmodified)
//     ===CODEWORDS_FILE===
//     <raw contents of Result_codewords.txt, written by the library into
//      the process's current working directory, if COUNT > 0 - see
//      LinCodeWeightInv's own user manual for its exact layout>
//     ===END===
//   mode "equal":
//     Same output shape as "less_than", but COUNT is the number of
//     codewords of weight EXACTLY W (via calculate_number_of_words_
//     with_fixed_w), and Result_codewords.txt (dumped the same way)
//     contains only codewords of weight exactly W - e.g. a planted
//     codeword of a known target weight, when "less_than" legitimately
//     returns 0 because nothing lighter exists.
//
// Build (from Src/testProgram/, AFTER the library itself has already been
// built with `make && make install` (or `make -f Makefile_neon && make -f
// Makefile_neon install` on Apple Silicon) from Src/ - see the project's
// own build instructions first). That produces ../lib/LinCodeWeightInv.a:
//   g++ -std=c++17 -O2 -march=native -I../include \
//       -o lincode_bridge LinCodeBridge.cpp ../lib/LinCodeWeightInv.a
//
// (On Apple Silicon, drop -march=native - it's an x86-only flag and will
// fail to compile on arm64; the Makefile_neon build already targets NEON
// instead. If you used CMake rather than the plain Makefile, the static
// library is named ../lib/libv1.3.a instead - substitute that filename.)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include "LinCodeWeightInv.h"

static int** read_matrix_file(const char* path, int& n, int& k, int& q) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open matrix file '%s'\n", path);
        exit(1);
    }
    if (fscanf(fp, "%d %d %d", &n, &k, &q) != 3) {
        fprintf(stderr, "ERROR: expected 'n k q' on the first line of '%s'\n", path);
        exit(1);
    }
    int** M = new int*[k];
    for (int i = 0; i < k; i++) {
        M[i] = new int[n];
        for (int j = 0; j < n; j++) {
            if (fscanf(fp, "%d", &M[i][j]) != 1) {
                fprintf(stderr,
                    "ERROR: expected k*n=%d integers (k=%d rows of n=%d), "
                    "ran out while reading row %d, col %d\n", k * n, k, n, i, j);
                exit(1);
            }
        }
    }
    fclose(fp);
    return M;
}

static void free_matrix(int** M, int k) {
    for (int i = 0; i < k; i++) delete[] M[i];
    delete[] M;
}

static void dump_file_if_exists(const char* path) {
    std::ifstream in(path);
    if (!in) {
        printf("(no %s produced - 0 codewords found)\n", path);
        return;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    fputs(ss.str().c_str(), stdout);
}

int main(int argc, char** argv) {
    detect();  // sets instructionSet from the running CPU's feature bits

    if (argc < 3) {
        fprintf(stderr,
            "Usage:\n"
            "  %s spectrum  <matrix_file>\n"
            "  %s less_than <matrix_file> <W>\n"
            "  %s equal     <matrix_file> <W>\n", argv[0], argv[0], argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    const char* matrix_path = argv[2];

    int n = 0, k = 0, q = 0;
    int** M = read_matrix_file(matrix_path, n, k, q);

    if (mode == "spectrum") {
        calculateWeightDistribution(M, n, k, q, /*multiplicativeForm=*/false);
        printf("===WEIGHTS===\n");
        for (int i = 0; i <= n; i++) {
            printf("%d,%llu\n", i, weights[i]);
        }
        printf("===END===\n");
    }
    else if (mode == "less_than") {
        if (argc < 4) {
            fprintf(stderr, "less_than mode requires <W>\n");
            free_matrix(M, k);
            return 1;
        }
        int W = atoi(argv[3]);
        unsigned long long count = calculate_number_of_words_less_than_fixed_w(
            M, n, k, q, W, /*write=*/true, /*multiplicativeForm=*/false);
        printf("COUNT:%llu\n", count);
        printf("===CODEWORDS_FILE===\n");
        dump_file_if_exists("Result_codewords.txt");
        printf("===END===\n");
    }
    else if (mode == "equal") {
        if (argc < 4) {
            fprintf(stderr, "equal mode requires <W>\n");
            free_matrix(M, k);
            return 1;
        }
        int W = atoi(argv[3]);
        unsigned long long count = calculate_number_of_words_with_fixed_w(
            M, n, k, q, W, /*write=*/true, /*multiplicativeForm=*/false);
        printf("COUNT:%llu\n", count);
        printf("===CODEWORDS_FILE===\n");
        dump_file_if_exists("Result_codewords.txt");
        printf("===END===\n");
    }
    else {
        fprintf(stderr, "Unknown mode '%s' (expected 'spectrum', 'less_than', or 'equal')\n", mode.c_str());
        free_matrix(M, k);
        return 1;
    }

    free_matrix(M, k);
    return 0;
}
