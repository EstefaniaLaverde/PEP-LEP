#include <time.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "testDriver.h"
#include "LinCodeWeightInv.h"
using namespace std;


int main(int argc, char** argv) {
	
	int flag = 0;
	int k = 3 , n = 500, q = 25, w_searched = 480, count = 0;
	unsigned long long int d = 0;
	bool found = false, multiplicative = false, write_CW = true;


	// ------------------------------User interface program-----------------------------------//

	char write_in_file = false;
	while (flag!=4) {
		printf("Please choose the input data:\n");
		printf("1. Reading generator matrix from file\n");
		printf("2. Generating random linear code\n");
		printf("3. Test Driver\n");
		printf("4. Exit\n");
		cin >> flag;
		if (flag == 1) {
			cout << "Enter file name: ";
			string in = "";
			char* fileName;
			cin >> in;
			fileName = new char[in.length() + 1];
			strcpy(fileName, in.c_str());
			int num_of_input = 0;
			int function = 0;
			while (function < 7) {
				printf("Please choose an option:\n");
				printf("1 Calculating weight spectrum\n");
				printf("2 Calculating minimum distance of code\n");
				printf("3 Searching for word with weight less than:\n");
				printf("4 Searching for word with weight equal to:\n");
				printf("5 Calculating number of words with weight less than:\n");
				printf("6 Calculating number of words with weight equal to\n");
				printf("7 Back\n");
				cin >> function;
				switch (function) {
				case 1:
					calculateWeightDistribution(fileName);
					break;
				case 2:
					min_dis(fileName);

					break;
				case 3:
					cout << "Enter searched weight: ";
					cin >> w_searched;
					find_word_less_than_fixed_weight(fileName, w_searched);
					break;
				case 4:
					cout << "Enter searched weight: ";
					cin >> w_searched;
					find_word_equal_to_fixed_weight(fileName, w_searched);
					break;
				case 5:
					cout << "Enter searched weight: ";
					cin >> w_searched;
					write_in_file = false;
					printf("Do you want to write the codewords in file?(y/n)\n");
					cin >> write_in_file;
					if (write_in_file == 'Y' || write_in_file == 'y') {
						calculate_number_of_words_less_than_fixed_w(fileName, w_searched, true);
					}
					else {
						calculate_number_of_words_less_than_fixed_w(fileName, w_searched, false);
					}
					break;
				case 6:
					cout << "Enter searched weight: ";
					cin >> w_searched;
					printf("Do you want to write the codewords in file?(y/n)\n");
					cin >> write_in_file;
					if (write_in_file == 'Y' || write_in_file == 'y') {
						calculate_number_of_words_with_fixed_w(fileName, w_searched, true);
					}
					else {
						calculate_number_of_words_with_fixed_w(fileName, w_searched, false);
					}
					break;
				case 7:
					break;
				default:
					printf("Wrong input\n");
					break;
				}
			}
		}
		else if (flag == 2) {

			cout << "Enter values for n,k,q" << endl;
			cin >> n >> k >> q;
			int function = 0;
			printf("Please choose an option:\n");
			printf("1 Calculating weight spectrum\n");
			printf("2 Calculating minimum distance of code\n");
			printf("3 Searching for word with weight less than:\n");
			printf("4 Searching for word with weight equal to:\n");
			printf("5 Calculating number of words with weight less than:\n");
			printf("6 Calculating number of words with weight equal to\n");
			cin >> function;
			switch (function) {
			case 1:
				calculateWeightDistribution(n, k, q);
				for (int i = 1; i <= n; i++) {
					if (weights[i] > 0) {
						printf("%d^%llu ", i, weights[i]);
						//cout << i << "^" << weights[i] << "  ";
					}
				}
				printf("\n");
				break;
			case 2:
				d = min_dis(n, k, q);
				printf("d = %llu\n", d);
				break;
			case 3:
				cout << "Enter searched weight: ";
				cin >> w_searched;
				found = find_word_less_than_fixed_weight(n, k, q, w_searched);
				if (found) {
					printf("Found a word with weight less than %d\n", w_searched);
				}
				else {
					printf("Did NOT find a word with weight less than %d\n", w_searched);
				}
				break;
			case 4:
				cout << "Enter searched weight: ";
				cin >> w_searched;
				found = find_word_equal_to_fixed_weight(n, k, q, w_searched);
				if (found) {
					printf("Found a word with weight = %d\n", w_searched);
				}
				else {
					printf("Did NOT find a word with weight = %d\n", w_searched);
				}
				break;
			case 5:
				cout << "Enter searched weight: ";
				cin >> w_searched;
				write_in_file = false;
				printf("Do you want to write the codewords in file?(y/n)\n");
				cin >> write_in_file;
				if (write_in_file == 'Y' || write_in_file == 'y') {
					unsigned long long int num = calculate_number_of_words_less_than_fixed_w(n, k, q, w_searched, true);
					printf("Found %llu words with weight less than %d\n", num, w_searched);
				}
				else {
					unsigned long long int num = calculate_number_of_words_less_than_fixed_w(n, k, q, w_searched, false);
					printf("Found %llu words with weight less than %d\n", num, w_searched);
				}
				break;
			case 6:
				cout << "Enter searched weight: ";
				cin >> w_searched;
				write_in_file = 'n';
				printf("Do you want to write the codewords in file?(y/n)\n");
				cin >> write_in_file;
				if (write_in_file == 'Y' || write_in_file == 'y') {
					unsigned long long int num = calculate_number_of_words_with_fixed_w(n, k, q, w_searched, true);
					printf("Found %llu words with weight = %d\n", num, w_searched);
					
				}
				else {
					unsigned long long int num = calculate_number_of_words_with_fixed_w(n, k, q, w_searched, false);
					printf("Found %llu words with weight = %d\n", num, w_searched);
				}
				break;
			default:
				printf("Wrong input\n");
				break;
			}

		}
		else if(flag == 3){
			test_drive();
		}
		else if (flag == 4) {
			break;
		}
		else {
			printf("Not an option\n\nPlease choose again\n");
		}
		printf("\n\n\n");

	}
	
	
	return 0;
}
