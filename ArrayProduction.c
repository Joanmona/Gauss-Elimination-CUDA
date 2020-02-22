#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

int *out_data;

int main(int argc , char *argv[])
{
	
		printf ("The number of arguments are %d \n", argc);
	
		for(int i=0;i<argc;i++)
		{
			printf("%s\n",argv[i]);
		}


		int lines = atoi(argv[1]); // για να τα κανει σε int
		int cols = atoi(argv[1]);
		cols=cols+1; // για την επιπλέον στήλη, τη σταθερά στήλη
		int size = lines*cols;
		
		srand ( time(NULL) ); // για να βγαζει διαφορετικους πινακες με το ιδιο μεγεθος (αλλα πρεπει να περασει λιγος χρονος)

		// Allocate the host output(final) 2D Array   
		out_data = (int*)malloc(sizeof(int*)*size);
	 
	 
	
		//δημιουργία του πίνακα με random αριθμούς από το 0 έως το 100
		for(int i=0;i<size;i++)
			{
	
				out_data[i] = rand() % 101;
		
			}
	
  
	
		FILE *final;
		final= fopen("pinakas.txt","w");

		if(final == NULL)
		{
			printf("Error!");   
               
		}

		for(int a=0; a<lines;a++)
		{
			fprintf(final, "[");
			for( int b=0;b<cols;b++)
			{
				int c = b+a*cols;
				int here = out_data[c] % 2; // mod 2  τον κάθε αριθμό για να δώσει είτε 0 είτε 1
				fprintf(final, "%d", here);
				// printf("c=%d \n",c);
				//Για πίνακα χωρίς κενά με τις αγκίλες
				if( b != cols-1)
				{
					fprintf(final, " ");
				}

				////printf("pin[%d][%d] = %d \n", a,b,pinakas[a][b]);
		 
		   
			}
	   
			fprintf(final, "]");
			fprintf(final, "\n");
		}
	
		fclose(final);
	
		free(out_data);

		return 0;

}
