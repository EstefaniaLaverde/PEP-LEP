CFLAGS =  -m64 -O3 -DNDEBUG -ansi -static-libstdc++ -static-libgcc -std=c++11 -march=native -I./include 
#If the target architecture has AVX512 VPOPCNTDQ instructions add the following flag to CFLAGS
#-mavx512vpopcntdq
CC = g++
#Alternative compilers:
#clang++
#x86_64-w64-mingw32-g++
BUILDDIR = build
OBJ = Test-UI

all: $(OBJ) clean

$(OBJ): driver.o test128_WeightDistribution.o test256_WeightDistribution.o test512_WeightDistribution.o testScalar_WeightDistribution.o \
test128_MinimumDistance.o test256_MinimumDistance.o test512_MinimumDistance.o testScalar_MinimumDistance.o \
test128_SearchEqual.o test256_SearchEqual.o test512_SearchEqual.o testScalar_SearchEqual.o \
test128_SearchLessThan.o test256_SearchLessThan.o test512_SearchLessThan.o testScalar_SearchLessThan.o \
test128_CountEqual.o test256_CountEqual.o test512_CountEqual.o testScalar_CountEqual.o \
test128_CountLessThan.o test256_CountLessThan.o test512_CountLessThan.o testScalar_CountLessThan.o \
 LinCodeWeightInv.a $(BUILDDIR)
	$(CC) $(CFLAGS) -lm driver.o -L. LinCodeWeightInv.a -o ./build/$(OBJ)
	
	
	$(CC) $(CFLAGS) -lm test128_WeightDistribution.o -L. LinCodeWeightInv.a -o ./build/test128_WeightDistribution
	$(CC) $(CFLAGS) -lm test256_WeightDistribution.o -L. LinCodeWeightInv.a -o ./build/test256_WeightDistribution
	$(CC) $(CFLAGS) -lm test512_WeightDistribution.o -L. LinCodeWeightInv.a -o ./build/test512_WeightDistribution
	$(CC) $(CFLAGS) -lm testScalar_WeightDistribution.o -L. LinCodeWeightInv.a -o ./build/testScalar_WeightDistribution
	
	$(CC) $(CFLAGS) -lm test128_MinimumDistance.o -L. LinCodeWeightInv.a -o ./build/test128_MinimumDistance
	$(CC) $(CFLAGS) -lm test256_MinimumDistance.o -L. LinCodeWeightInv.a -o ./build/test256_MinimumDistance
	$(CC) $(CFLAGS) -lm test512_MinimumDistance.o -L. LinCodeWeightInv.a -o ./build/test512_MinimumDistance
	$(CC) $(CFLAGS) -lm testScalar_MinimumDistance.o -L. LinCodeWeightInv.a -o ./build/testScalar_MinimumDistance
	
	$(CC) $(CFLAGS) -lm test128_SearchEqual.o -L. LinCodeWeightInv.a -o ./build/test128_SearchEqual
	$(CC) $(CFLAGS) -lm test256_SearchEqual.o -L. LinCodeWeightInv.a -o ./build/test256_SearchEqual
	$(CC) $(CFLAGS) -lm test512_SearchEqual.o -L. LinCodeWeightInv.a -o ./build/test512_SearchEqual
	$(CC) $(CFLAGS) -lm testScalar_SearchEqual.o -L. LinCodeWeightInv.a -o ./build/testScalar_SearchEqual
	
	$(CC) $(CFLAGS) -lm test128_SearchLessThan.o -L. LinCodeWeightInv.a -o ./build/test128_SearchLessThan
	$(CC) $(CFLAGS) -lm test256_SearchLessThan.o -L. LinCodeWeightInv.a -o ./build/test256_SearchLessThan
	$(CC) $(CFLAGS) -lm test512_SearchLessThan.o -L. LinCodeWeightInv.a -o ./build/test512_SearchLessThan
	$(CC) $(CFLAGS) -lm testScalar_SearchLessThan.o -L. LinCodeWeightInv.a -o ./build/testScalar_SearchLessThan
	
	$(CC) $(CFLAGS) -lm test128_CountEqual.o -L. LinCodeWeightInv.a -o ./build/test128_CountEqual
	$(CC) $(CFLAGS) -lm test256_CountEqual.o -L. LinCodeWeightInv.a -o ./build/test256_CountEqual
	$(CC) $(CFLAGS) -lm test512_CountEqual.o -L. LinCodeWeightInv.a -o ./build/test512_CountEqual
	$(CC) $(CFLAGS) -lm testScalar_CountEqual.o -L. LinCodeWeightInv.a -o ./build/testScalar_CountEqual
	
	$(CC) $(CFLAGS) -lm test128_CountLessThan.o -L. LinCodeWeightInv.a -o ./build/test128_CountLessThan
	$(CC) $(CFLAGS) -lm test256_CountLessThan.o -L. LinCodeWeightInv.a -o ./build/test256_CountLessThan
	$(CC) $(CFLAGS) -lm test512_CountLessThan.o -L. LinCodeWeightInv.a -o ./build/test512_CountLessThan
	$(CC) $(CFLAGS) -lm testScalar_CountLessThan.o -L. LinCodeWeightInv.a -o ./build/testScalar_CountLessThan
	
	
lib512.o: ./src/lib512.cpp ./include/lib512.h ./include/lib256.h ./include/lib128.h ./include/ReadWrite.h ./include/Data.h ./include/Polynomials.h
	$(CC) $(CFLAGS) -c ./src/lib512.cpp -o lib512.o
lib256.o: ./src/lib256.cpp ./include/lib256.h ./include/lib128.h ./include/ReadWrite.h ./include/Data.h ./include/Polynomials.h
	$(CC) $(CFLAGS) -c ./src/lib256.cpp -o lib256.o
lib128.o: ./src/lib128.cpp ./include/lib128.h ./include/ReadWrite.h ./include/Data.h ./include/Polynomials.h
	$(CC) $(CFLAGS) -c ./src/lib128.cpp -o lib128.o
Data.o: ./src/Data.cpp ./include/Data.h
	$(CC) $(CFLAGS) -c ./src/Data.cpp -o Data.o
Polynomials.o: ./src/Polynomials.cpp ./include/Polynomials.h
	$(CC) $(CFLAGS) -c ./src/Polynomials.cpp -o Polynomials.o
ReadWrite.o: ./src/ReadWrite.cpp ./include/ReadWrite.h ./include/Polynomials.h
	$(CC) $(CFLAGS) -c ./src/ReadWrite.cpp -o ReadWrite.o
LinCodeWeightInv.o: ./src/LinCodeWeightInv.cpp ./include/LinCodeWeightInv.h ./include/ReadWrite.h ./include/Polynomials.h ./include/Data.h ./include/lib128.h ./include/lib256.h ./include/lib512.h
	$(CC) $(CFLAGS) -c ./src/LinCodeWeightInv.cpp -o LinCodeWeightInv.o
testDriver.o: ./src/testDriver.cpp ./include/testDriver.h ./include/LinCodeWeightInv.h ./include/Data.h
	$(CC) $(CFLAGS) -c ./src/testDriver.cpp -o testDriver.o
driver.o: ./testProgram/Source.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Source.cpp -o driver.o
	
	
test128_WeightDistribution.o: ./testProgram/Test128.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test128.cpp -o test128_WeightDistribution.o
test256_WeightDistribution.o: ./testProgram/Test256.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test256.cpp -o test256_WeightDistribution.o
test512_WeightDistribution.o: ./testProgram/Test512.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test512.cpp -o test512_WeightDistribution.o
testScalar_WeightDistribution.o: ./testProgram/TestScalar.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/TestScalar.cpp -o testScalar_WeightDistribution.o
	
	
test128_MinimumDistance.o: ./testProgram/Test128_MinimumDistance.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test128_MinimumDistance.cpp -o test128_MinimumDistance.o
test256_MinimumDistance.o: ./testProgram/Test256_MinimumDistance.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test256_MinimumDistance.cpp -o test256_MinimumDistance.o
test512_MinimumDistance.o: ./testProgram/Test512_MinimumDistance.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test512_MinimumDistance.cpp -o test512_MinimumDistance.o
testScalar_MinimumDistance.o: ./testProgram/TestScalar_MinimumDistance.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/TestScalar_MinimumDistance.cpp -o testScalar_MinimumDistance.o
	

test128_SearchEqual.o: ./testProgram/Test128_SearchEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test128_SearchEqual.cpp -o test128_SearchEqual.o
test256_SearchEqual.o: ./testProgram/Test256_SearchEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test256_SearchEqual.cpp -o test256_SearchEqual.o
test512_SearchEqual.o: ./testProgram/Test512_SearchEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test512_SearchEqual.cpp -o test512_SearchEqual.o
testScalar_SearchEqual.o: ./testProgram/TestScalar_SearchEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/TestScalar_SearchEqual.cpp -o testScalar_SearchEqual.o
	
test128_SearchLessThan.o: ./testProgram/Test128_SearchLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test128_SearchLessThan.cpp -o test128_SearchLessThan.o
test256_SearchLessThan.o: ./testProgram/Test256_SearchLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test256_SearchLessThan.cpp -o test256_SearchLessThan.o
test512_SearchLessThan.o: ./testProgram/Test512_SearchLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test512_SearchLessThan.cpp -o test512_SearchLessThan.o
testScalar_SearchLessThan.o: ./testProgram/TestScalar_SearchLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/TestScalar_SearchLessThan.cpp -o testScalar_SearchLessThan.o
		
	
test128_CountEqual.o: ./testProgram/Test128_CountEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test128_CountEqual.cpp -o test128_CountEqual.o
test256_CountEqual.o: ./testProgram/Test256_CountEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test256_CountEqual.cpp -o test256_CountEqual.o
test512_CountEqual.o: ./testProgram/Test512_CountEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test512_CountEqual.cpp -o test512_CountEqual.o
testScalar_CountEqual.o: ./testProgram/TestScalar_CountEqual.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/TestScalar_CountEqual.cpp -o testScalar_CountEqual.o
	
test128_CountLessThan.o: ./testProgram/Test128_CountLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test128_CountLessThan.cpp -o test128_CountLessThan.o
test256_CountLessThan.o: ./testProgram/Test256_CountLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test256_CountLessThan.cpp -o test256_CountLessThan.o
test512_CountLessThan.o: ./testProgram/Test512_CountLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/Test512_CountLessThan.cpp -o test512_CountLessThan.o
testScalar_CountLessThan.o: ./testProgram/TestScalar_CountLessThan.cpp ./include/LinCodeWeightInv.h ./include/testDriver.h
	$(CC) $(CFLAGS) -c ./testProgram/TestScalar_CountLessThan.cpp -o testScalar_CountLessThan.o
	
LinCodeWeightInv.a: lib128.o lib256.o lib512.o  ReadWrite.o Data.o LinCodeWeightInv.o testDriver.o Polynomials.o
	ar rcs LinCodeWeightInv.a lib128.o lib256.o lib512.o  ReadWrite.o Data.o LinCodeWeightInv.o testDriver.o Polynomials.o

libs: LinCodeWeightInv.a
$(BUILDDIR): ./testFiles/TestDataBig ./testFiles/TestDataSmall ./testFiles/ArticleData 
	mkdir -p build
	cp ./testFiles/TestDataBig ./build/TestDataBig
	cp ./testFiles/TestDataSmall ./build/TestDataSmall
	cp ./testFiles/ArticleData ./build/ArticleData
install: LinCodeWeightInv.a
	mkdir -p lib
	cp LinCodeWeightInv.a ./lib/LinCodeWeightInv.a
	-rm *.o
clean:
	-rm *.o

