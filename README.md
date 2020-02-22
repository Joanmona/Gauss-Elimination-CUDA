# Gauss-Elimination-CUDA
Parallel implemetation of Gauss Elimination (thesis)

Way to run the program.

1. For random array
1.1 gcc ArrayProduction.c -o ArrayProduction (-std=c99 για cc=1.x)
1.2 ./ArrayProduction x (x is the size of the array, x x (x+1) )

2. Gauss Reduction
2.1 gcc -c Read_And_Write_Linear.c -lgmp (-std=c99 για cc=1.x)
2.2 nvcc -c -arch=sm_x Ptixiaki_GPU_Loops_Kernels_s_s.cu 
2.3 nvcc Read_And_Write_Linear.o
2.4 ./a.out pinakas.txt 2

3. Backtracking - solutions
3.1 gcc backwards.c Read_And_Write_Linear.c -o back (-std=c99 για cc=1.x)
3.2 ./back final.txt
