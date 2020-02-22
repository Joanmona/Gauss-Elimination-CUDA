#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h> 


////CUDA uses a C++ compiler to compile .cu files. Therefore, it expects that all functions referenced in .cu files
////have C++ linkage unless explicitly instructed otherwise. 
////And in my case, I must explicitly instruct the C++ compiler otherwise.

extern "C" { 
#include "Read_And_Write_Linear.h"
}


#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>

#include "cuPrintf.cu"

//για τον έλεγχο των σφαλμάτων
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
} 

/////////////////////
/////// Parallel In GPU
////////////////

int *in_data;           //store array in CPU
int *in_gpu_data;  //store array in GPU

int *lines_one, *gpu_lines_one; // Χρήση για την εύρεση της γραμμής που θα χρησιμοποιθεί στην xor
int *lines_zeros, *gpu_lines_zeros; // Χρήση για το μέτρημα των μηδενικών

clock_t start, end;  // Μετράνε το χρόνο εκτέλεσης του προγράμματος
double cpu_time_used; // Ο τελικός χρόνος εκτέλεσης του προγράμματος




/*
 * Η Συνάρτηση αυτή εκτελείται από τη GPU και κάνει swap τις γραμμές που έχουμε επιλέξει.
 * Δέχεται σαν παραμέτρους τον πίνακα μετά από κάθε "κύκλο" του Gauss, τον αριθμό των στηλών, των γραμμών,τη θέση pivot (at),
 * τη στήλη στην οποία βρισκόμαστε, τη γραμμή στην οποία βρισκόμαστε και τη γραμμή με την οποία θα γίνει 
 * η ανταλλαγή (swap_line). 
 * Κάθε thread είναι υπεύθυνο να κάνει swap τη θέση της γραμμής του pivot με την αντίστοιχη θέση στη swap_line και είναι στην
 * ίδια στήλη για την οποία είναι υπεύθυνο. 
 * 
 */

__global__ void swap(int *in_gpu_data, int cols, int lines, int at, int start_col, int start_line, int swap_line)
{

		int i = blockDim.x * blockIdx.x + threadIdx.x;  //global id
		//printf(" i = %d \n",i);
		//printf(" blockDim.x = %d \n",blockDim.x);
		//printf(" blockIdx.x = %d \n",blockIdx.x );
		//printf(" threadIdx.x = %d \n",threadIdx.x);
		
       if (i==1)
			{
				cuPrintf ( " SWAP \n ");
			}
   
    
		__syncthreads(); // Για να ξυπνήσουν όλα τα threads 

		
		      
    	// τα αχρηστα/περισευούμενα τα threads να μη δουλευουν
	    // ο αριθμός των threads που θα χρειαστούν είναι ίσο με τον αριθμό των στηλών που θα γίνει ανταλλαγή
		//Δηλαδή από το pivot και μετά. 
				  
                      
		if (i<(cols-start_col))
			{
					   
				//cuPrintf(" iii = %d \n", i);
				
			    //Κάθε thread είναι υπεύθυνο για την ανταλλαγή των 2 θέσεων που βρίσκονται στην ίδια στήλη.
                       
					        
			    int h = swap_line-start_line;
			    
			     //at είναι η θεση στην οποία βρισκόμαστε, και οι υπόλοιπες θέσεις που θα ανταλλαχτούν (a1) από 
			     //τη γραμμή στην οποία βρισκόμαστε (η γραμμή που έχει pivot=0) βρίσκονται πάνω στην ίδια γραμμή
			     //αλλά σε διαφορετικές στήλες. Για το at είναι υπεύθυνο το thread 0, για την επόμενη θεση (at+1)
			     //είναι υπεύθυνο το thread 1 κτλ.
			    int a1 = at + i;
               
                  // Το a2 είναι οι θέσεις της γραμμής με την οποία θα ανταλλαχτεί η πάνω γραμμή. Η κάθε θέση
                  //βρίσκεται στην ίδια στήλη με τη θέση που θα κάνει ανταλλαγή αλλά μερικές γραμμές κάτω.
                 //Το h καθορίζει πόσες γραμμές κάτω είναι, η διαφορά τους σε μία γραμμή
                 //είναι όσο ο αριθμός των στηλών.
               int a2 = at + cols*h + i;
                            
			   //cuPrintf(" a1 i= %d %d\n", a1,i);
			   //cuPrintf(" a2 i= %d %d\n", a2,i);
                     
               //Κάνε ανταλλαγή τις θέσεις
			   int tt = in_gpu_data[a1];
			   in_gpu_data[a1] = in_gpu_data[a2];
			   in_gpu_data[a2] = tt;

		    }
   
		__syncthreads();
				  

}



/*
 * Η Συνάρτηση αυτή εκτελείται από τη GPU και κάνει xor τις γραμμές που έχουν 1 στην ίδια στήλη που βρίσκεται το pivot.
 * Δέχεται σαν παραμέτρους τον πίνακα μετά από κάθε "κύκλο" του Gauss, τον αριθμό των στηλών, των γραμμών,τη θέση pivot (at),
 * τη στήλη στην οποία βρισκόμαστε, τη γραμμή στην οποία βρισκόμαστε και τον πίνακα gpu_lines_ones που περιέχει τον αριθμό 1
 * στις θέσεις όπου οι αντίστοιχες γραμμές έχουν 1 στην ίδια στήλη με το pivot. 
 * Κάθε thread είναι υπεύθυνο να κάνει xor τη θέση της γραμμής του pivot με τις θέσεις που ανήκουν στη στήλη 
 * για την οποία είναι υπεύθυνα για όλες τις γραμμές που έχουν 1 στον πίνακα gpu_lines_one. 
 * 
 */

__global__ void xor_(int *in_gpu_data, int cols, int lines,int at, int start_col, int start_line, int *gpu_lines_one)
{

         int i = blockDim.x * blockIdx.x + threadIdx.x; 
	    //printf(" i = %d \n",i);
		//printf(" blockDim.x = %d \n",blockDim.x);
		//printf(" blockIdx.x = %d \n",blockIdx.x );
		//printf(" threadIdx.x = %d \n",threadIdx.x);
     
        if (i==1)
          {
           cuPrintf ( " XOR \n ");
          }
    
	    
		   __syncthreads(); // Για να ξυπνήσουν όλα τα threads 
		
		
			  
		int perisema = cols-start_col;
			  
		//τα threads που χρειάζονται είναι όσα είναι οι στήλες που θα κάνουν xor, δηλαδή από τη στήλη του pivot 
		//(μαζί με αυτή του pivot) και πέρα
		
	    if( i>=0 && i<perisema) //Θέλω να κάνω xor και τη στήλη στην οποία είμαι (start_col)
			  	 {
	    	
				  // cuPrintf("perisema i = %d %d\n", perisema,i);
				  //cuPrintf("pcols i = %d %d\n", cols,i);
				  //cuPrintf("pstart i = %d %d\n", start_col,i);
	    	
	    	
				  int at1 = at + i;// Οι θέσεις που θα γίνουν xor αλλά δε θα αλλάξουν κινούνται πάνω στην ίδια γραμμή 
				  //cuPrintf("at1 i = %d %d\n",at1,i);
				  
				  for ( int f=0; f < lines ; f++ ) // για κάθε γραμμή, βρες ποιες γραμμές έχουν 1 κάτω από το pivot
				  	  {
				          
					     if (gpu_lines_one[f] == 1) // αν η θέση έχει 1 τότε κάνε xor με την αντίστοιχη γραμμή
					     {
					    	 int grammh = f;
					    	 int jumps = grammh - start_line; // πόσες γραμμές θα πρέπει να κατέβει για να πάει στη γραμμή με την οποία θα κάνει xor
					    	 int at2 = at1 + cols*jumps; //θέση με την οποία θα κάνει xor. Βρίσκεται στην ίδια στήλη αλλά γραμμές πιο κάτω.
					    	 //cuPrintf("at2 i = %d %d\n",at2,i);
					    	 
					    	 //κάνε xor 
					    	 in_gpu_data[at2]= in_gpu_data[at2] ^ in_gpu_data[at1];	 
					    	 
					     } // τελος if
				  

				  	  } // τελος for
			  
			  	  }//τελος if
			  
			  
			  __syncthreads(); 
			
}




/*
 * Η Συνάρτηση αυτή εκτελείται από τη GPU και βρίσκει τα μηδενικά που βρίσκονται κάτω από τη διαγώνιο.
 * Δέχεται σαν παραμέτρους τον πίνακα μετά από κάθε "κύκλο" του Gauss, τον αριθμό των στηλών, των γραμμών
 * και έναν πίνακα (gpu_lines_zeros) στον οποίο θα αποθηκευτεί ο αριθμός των μηδενικών κάθε στήλης.
 * Κάθε thread είναι υπεύθυνο να βρει τον αριθμό των μηδενικών της στήλης που είναι υπεύθυνα και να
 * αποθηκεύσει τον αριθμό αυτό στην αντίστοιχη θέση στον πίνακα gpu_lines_zeros πχ τα μηδενικά της
 * στήλης 0 θα αποθηκευτούν στη gpu_lines_zeros[0].
 * 
 */

__global__ void find_zeros(int *in_gpu_data, int cols, int lines, int *gpu_lines_zeros)
{

         int i = blockDim.x * blockIdx.x + threadIdx.x; 
	    //printf(" i = %d \n",i);
		//printf(" blockDim.x = %d \n",blockDim.x);
		//printf(" blockIdx.x = %d \n",blockIdx.x );
		//printf(" threadIdx.x = %d \n",threadIdx.x);
     
       if (i==1)
        {
           cuPrintf ( " ZEROS\n ");
        }
    
	      // printf("size = %d\n", size);
		   __syncthreads(); // Για να ξυπνήσουν όλα τα threads 
		
	
		   
		   if(i>=0 && i<cols-2)
		     {
			   int jump = cols+1;
			  // cuPrintf ("jump= %d\n",jump);
			   
			   // Η αρχική μου θέση είναι η πρώτη στήλη
			   // Κάθε thread αντιστοιχίζεται με τη στήλη που έχει τον ίδιο αριθμό με το global ID τους.
			   // Τα threads απλώνονται μέχρι και την προ-τελευταία στήλη (χωρίς αυτήν)
			  
			   int at1 = jump*i; //Θέση πάνω στη διαγώνιο για κάθε στήλη
			  // cuPrintf("start at1 = %d \n",at1);
			   
			   int at2 = at1 + cols; // Ο έλεγχος θα ξεκινήσει από τη θέση πάτω από τη διαγώνιο
			   //cuPrintf("katw thesh at2 = %d \n",at2);
			   
			   for(int t=i+1; t<lines; t++) // για να φτάσει μέχρι και την προ-προ τελευταία στήλη
			   {
				   if(in_gpu_data[at2] == 0) // αν έχει 0 τότε μέτρα το
				   {
					   gpu_lines_zeros[i]= gpu_lines_zeros[i]+1; // τα μηδενικά σε κάθε στήλη
				   }
				   
				   at2 = at2+cols;//κατέβα στην ακριβώς από κάτω θέση (ίδια στήλη επόμενη γραμμή)
				   
			   }
			   		   		   
		   }
		   
		   
		   
		   __syncthreads(); 
		  
			  
}




////////////////////////////////////////////////////////////////////////////////
//////////////////////////// MAIN /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


	 
int main(int argc , char *argv[])
{
	
	printf ("The number of arguments are %d \n", argc);
	
	for(int i=0;i<argc;i++)
	  {
		printf("%s\n",argv[i]);
	  }
		
		
	 
  
	 // data is a 1D array where the gauss array will be saved, already allocated into the host, it's been allocated
     in_data = ReadFile(argv[1]); 
	
     /*
     for(int u=0; u<info.Size;u++)
     {
    	 printf("d[%d] = %d \n",u,in_data[u]);
     }
     */
	printf("number of lines %d \n", info.lines);
	printf("number of cols %d \n", info.cols);
	printf("size %d \n", info.Size);
	
		
	printf("END \n");
	
	
	
	
	
	 /*
        //start = clock();
        cudaEvent_t time1,time2;
        cudaEventCreate(&time1);
        cudaEventCreate(&time2);
	
        cudaEventRecord(time1);
	 */

	
	      //Πίνακας που θα αποθηκεύει 1 σε όποια γραμμή έχει 1 στην ίδια στήλη με το pivot. Aρχικοποίηση με 0.
		  lines_one = (int*)malloc(sizeof(int*)*info.lines);
			
	      for(int h=0; h< info.lines ; h++)
	      {
	         lines_one[h]=0;
	      }
	      printf("END 6\n");
	 
	      
	      //Πίνακας που θα αποθηκεύει τον αριθμό των μηδενικών κάθε στήλης 
	      // στην αντίστοιχη θέση του. Aρχικοποίηση με 0.
	      lines_zeros = (int*)malloc(sizeof(int*)*info.lines);
 
	      for(int h2=0; h2< info.lines ; h2++)
	      	      {
	      	         lines_zeros[h2]=0;
	      	      }
	      printf("END 6.1\n");
    
	      
	      
    int h_start_col = 0; // αρχικοποίηση της στήλης που θα βρισκόμαστε κάθε φορά
	int h_start_line = 0; // αρχικοποίηση της γραμμής που θα βρισκόμαστε κάθε φορά
	int at = 0; // αρχικη θεση που ξεκιναμε για γραμμή και στήλη 0
	//int h_anw = 0;
	
	//Εύρεση του αριθμού των μηδενικών που πρέπει να υπάρχουν κάτω από τη διαγώνιο για να θεωρείται ο πίνακας άνω τριγωνικός
	int Anw_Trigwnikos = ((info.lines * info.lines)-info.lines)/2;
	printf( "number of zeros = %d \n", Anw_Trigwnikos);
  
	

	int k=0;
  
	printf("END 1\n");
  
	//Κάθε επανάληψη είναι ένας κύκλος κατά την εφαρμογή του Gauss.
	while ( k==0)
	{
  
       
        printf(" start_col ///// 1 = %d\n", h_start_col);
		printf( "start_line ////// 1 = %d\n", h_start_line);
  
  
		//  cudaMalloc
		// Allocate to the device memory for the array
	
		gpuErrchk( cudaMalloc((void **)&in_gpu_data, sizeof(int*)*info.Size));

		printf("END 2\n");
		
         
	
		// cudaMemcpy
	
		// Copy the host input (in_data) in host memory to the device input (in_gpu_data) device memory 
    
		printf("Copy input data from the host memory to the CUDA device\n");
	
		gpuErrchk(cudaMemcpy(in_gpu_data, in_data, sizeof(int*)*info.Size, cudaMemcpyHostToDevice));

		printf("END 3\n");
	
			
         
		 // at  η θεση στην οποια βρισκομστε κάθε φορά
			
		if(h_start_col ==  h_start_line) // αν η γραμμή στην οποία βρισκόμαστε ταυτιζεται με τη στήλη στην οποία βρισκόμαστε
			{
				if(h_start_col != 0) // και δεν είναι η γραμμή και στήλη 0
				{
					at = at + info.cols + 1; // τότε η επόμενη θέση pivot είναι η επόμεν θέση πάνω στη διαγώνιο
				}
			}
		else if (h_start_col > h_start_line) // αλλιώς αν ο αριθμός της στήλης που βρισκόμαστε είναι μεγαλύτερος από αυτόν της γραμμής
			{
				at = at + info.cols + 1; // τότε η επόμενη θέση είναι στην επόμενη γραμμή και επόμενη στήλη από αυτές στις οποίες βρισκόμαστε
			}
		
		//printf (".///// at = %d\n", at);
		//printf (" [%d,%d] = %d\n",h_start_line,h_start_col,in_data[at]);
	  
/**********************************************************************************************************************************/	  
	  
	  
	    printf("END 5\n");
	 
		// αν η θεση στην οποια βρισκομστε ειναι 0 θα πρέπει να βρούμε μια γραμμή που έχει 1 στην ίδια στήλη στην οποία βρισκόμαστε
		if ( in_data[at] == 0)
			{
	           	printf("END 5.1\n");
				
				int swap_line=0; // η γραμμή με την οποία θα κάνει ανταλλαγή / αρχικοποίηση με 0 
	  
				while ( swap_line == 0 )
					{
	  
	                    printf("END 5.2\n");
								
						// ευρεση της γραμμς που θα γινει η ανταλλαγή
	                    // αν δε βρει στην ίδια στήλη κάποια γραμμή με 0 τοτε θα συνεχίσει να ψάχνει στην ακριβώς επόμενη στήλη
	                    
	                      for( int i = 0; i < (info.lines-h_start_line); i++ ) 
							{
	                    	  
	                    	   int at2 = at + (info.cols * i) ; // οι θέσεις κάτω ακριβώς από τη pivot
		                       // printf(" t2 -= %d\n", at2);
							   if ( in_data[at2] == 1 ) //αν είναι 1 τότε κράτα τη γραμμή
									{
								        //printf(" t23 -= %d\n", at2);
										swap_line = h_start_line+i;
										break; //for
									}
	  
	  
							}
		 
		                  	printf("END 5.3\n");
		                  	
						  if ( swap_line == 0 ) // δηλαδή δε βρήκε γραμμή για να κάνει swap
							 {
							    printf("END 5.4\n");
		      
							    h_start_col++; //αλλαγή στήλης
							    at++; // πηγαίνει στη δίπλα θέση
							}
						  else
							{		
								break;//while
							}
							
						
					} // τελος while
	
	
				printf("END 5.4\n");
		 			 
				printf(" swp line = %d\n", swap_line);
				
				printf(" SWAP\n");
				cudaPrintfInit();
				// καλώ τον kernel swap_ για να κάνει την ανταλλαγή γραμμών παράλληλα
				swap<<< 47, 64 >>>(in_gpu_data, info.cols, info.lines, at, h_start_col, h_start_line,swap_line); 
				gpuErrchk(cudaGetLastError());
 
	 
	            printf("END 5.5\n");
	 
				gpuErrchk(cudaMemcpy(in_data, in_gpu_data, sizeof(int*)*info.Size, cudaMemcpyDeviceToHost));
				//^ σταματαει τη cpu και περιμενει τον kernel να τελειωσει για να παρει αποτελεσματα
				cudaPrintfDisplay(stdout,true);
				cudaPrintfEnd();
				gpuErrchk(cudaDeviceReset());  // αδειάζει τη gpu τελείως


                printf("END 5.6\n");

                //Επομένως πρέπει να ξανακάνει allocate ότι χρειάζεται
				// cudaMalloc
	
				gpuErrchk( cudaMalloc((void **)&in_gpu_data, sizeof(int*)*info.Size));

				printf("END 5.7\n");
		

				// cudaMemcpy
	
				// Copy the host input (in_data) in host memory to the device input (in_gpu_data)
				// device memory
    
				printf("Copy input data from the host memory to the CUDA device\n");
	
				gpuErrchk(cudaMemcpy(in_gpu_data, in_data, sizeof(int*)*info.Size, cudaMemcpyHostToDevice));

				printf("END 5.9\n");
	
			}	
			
		

	   
	  /*	 for(int u=0; u<info.Size;u++)
		     {
		    	 printf("d[%d] 2 = %d \n",u,in_data[u]);
		     }

	  */ 
	   
		
	   printf("END 6\n");
    /******************************************************************************************************************************/
		// XOR 
	   
	   printf("END 6.1\n");
	   //allocate memory for gpu_lines_one
	   gpuErrchk( cudaMalloc((void **)&gpu_lines_one, sizeof(int*)*info.lines));
	   
	   //Ψάχνει να βρεις ποιες γραμμές έχουν 1 στην ίδια στήλη με το pivot, κάτω από τη γραμμή του pivot.
	   //Όποια έχει 1 τότε θα μπει 1 στην αντίστοιχη θέση του πίνακα gpu_lines_one
	   //Αυτές οι γραμμές για γίνουν xor με τη γραμμή στην οποία ανήκει το pivot
	   
	   int perisema = info.lines-h_start_line;
	   //printf(" perisems = %d \n", perisema);
	  
	   int c_l=h_start_line;
	   //printf("c_l 1= %d \n",c_l);
	   
	   for(int f=1; f < perisema; f++)
	   {
		   //printf("f = %d \n", f);
		   c_l++;
		   int x= at+info.cols*f;// η θέση η οποία ελέγχεται κάτω από το pivot
		   //printf("x = %d \n",x);
		   
		   if (in_data[x]==1) 
		   {
			   //printf("ffffffffffffff \n");
			   //printf("cl = %d \n",c_l);
			   lines_one[c_l] = 1;
			  	   
		   }

	   }
	   
	 /*   for(int g=0;g<info.lines;g++)
	  	   { 
	  		   printf(" g  hF = %d %d\n",g,lines_one[g]);
	  		   
	  	   }
	  */ 
	  
	   printf("END 6.2\n");
	   //στειλε τον πίνακα στη gpu
	   gpuErrchk( cudaMemcpy(gpu_lines_one, lines_one, sizeof(int*)*info.lines, cudaMemcpyHostToDevice));

	   printf(" XOR\n");
	   cudaPrintfInit();
	   // Καλώ τον kernel xor για να κάνει xor όσες γραμμές έχουν 1 στη στήλη του pivot
		xor_<<<47, 64 >>>(in_gpu_data, info.cols, info.lines, at,h_start_col,h_start_line,gpu_lines_one);
		gpuErrchk(cudaGetLastError());

        printf("END 6.3\n");
      
        gpuErrchk(cudaMemcpy(in_data, in_gpu_data, sizeof(int*)*info.Size, cudaMemcpyDeviceToHost));
        //^ σταματαει τη cpu και περιμενει τον kernel να τελειωσει για να παρει αποτελεσματα
        cudaPrintfDisplay(stdout,true);
        cudaPrintfEnd();
		
		printf("END 6.4\n");
		
	/*	 for(int u=0; u<info.Size;u++)
		     {
		    	 printf("d[%d] 3 = %d \n",u,in_data[u]);
		     }
*/
		
	/********************************************************************************************************************************************************************/
		
	    /////// Ελεγχος για το αν ο πινακας εγινε ανω τριγωικός
           
		   //allocate memory to gpu, send the array as well (αρχικοποιημένο με 0)
		gpuErrchk( cudaMalloc((void **)&gpu_lines_zeros, sizeof(int*)*info.lines));
		gpuErrchk( cudaMemcpy(gpu_lines_zeros, lines_zeros, sizeof(int*)*info.lines, cudaMemcpyHostToDevice));
		  
		printf(" ZEROS\n");
		cudaPrintfInit();
		// καλώ τον kernel find_zeros για να βρει τα μηδενικά που έχει κάθε στήλη κάτω από τη διαγώνιο
		find_zeros<<<47, 64 >>>(in_gpu_data, info.cols, info.lines, gpu_lines_zeros);
		gpuErrchk(cudaGetLastError());
			  			
			  	    
		printf("END 6.3\n");
			  	      
		//παίρνω τον πίνακα με τα μηδενικά
		gpuErrchk(cudaMemcpy(lines_zeros, gpu_lines_zeros, sizeof(int*)*info.lines, cudaMemcpyDeviceToHost));
		 //^ σταματαει τη cpu και περιμενει τον kernel να τελειωσει για να παρει αποτελεσματα
		cudaPrintfDisplay(stdout,true);
		cudaPrintfEnd();
			  			
		printf("END 6.4\n");
               
		   
		int zeros=0;
		//μετράω πόσα είναι όλα τα μηδενικά
		for(int p=0;p<info.lines;p++)
			  {
				  zeros = zeros + lines_zeros[p];
				  
			  }
			
         
		// αν ο αριθμός των μηδενικών είναι ίσος με αυτόν που θέλουμε για να γίνει άνω τριγωνικός στμάτα τη while 
           if (zeros == Anw_Trigwnikos)
           {
        	   k=1;
           }
		
        printf("END 7\n");
        
        //αλλαγή γραμμής και στήλης σε περίπτωση που δεν είναι ανω τριγωνικός
        h_start_col++;
        h_start_line++;
        		
        printf(" start_col ///// 2= %d\n", h_start_col);
        printf( "start_line ////// 2= %d\n", h_start_line);
        		
        		
        printf("END 6.3\n");
        
    	
    	gpuErrchk(cudaDeviceReset());     

    	 for(int g=0;g<info.lines;g++)
    		   {
    			  //ξαναμηδενίζουμε τους πίνακες για να ξαναχρησιμοποιηθούν
    			  // printf(" g  hF = %d %d\n",g,lines_one[g]);
    			  lines_one[g]=0;
    			  lines_zeros[g]=0;
    			   
    		   }
    
    
  }
    
    printf("END 7.1\n");
    
   
    gpuErrchk(cudaPeekAtLastError());
   
	/*
	for(int i = 0; i < info.Size; i++)
    {
        printf("data[%d]=%d\n", i, out_data[i]);
    }
	printf("END 10 \n");
	
//////////////////////////////


cudaEventRecord(time2);

printf("END 10.1 \n");
    
    cudaEventSynchronize(time2);
    printf("END 10.2 \n");
    float totalTime =0;
    printf("END 10.3 \n");
    cudaEventElapsedTime(&totalTime, time1, time2);
    printf("END 10.4 \n");
	
	
*/

printf("END 10.4 \n");

WriteFile(in_data);

printf("END 10.5 \n");




    
    
   
    // Free host memory
    free(in_data);
    free(lines_one);
    free(lines_zeros);
    cudaProfilerStop();
    printf("END 11 \n");

   printf("Done\n");
  //printf("Total time elapsed = %5.2f ms\n", totalTime);
  // printf("Total time elapsed = %f \n", totalTime);
     

	 
	 
    return 0;

}// end of main


	 


	 
