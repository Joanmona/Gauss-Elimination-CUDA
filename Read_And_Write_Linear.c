#include <stdio.h>
#include <stdlib.h>
#include "Read_And_Write_Linear.h"

struct ArrayInfo info;




//////1. Function For Reading The Array For Gauss

/* 
 * This function reads the given array and allocates memory for it.
 * It also exracts the info of the array like its size and writes them into struck/header.
 */


int* ReadFile(char* File)
{
	
	FILE* file = fopen(File,"r"); //άνοιξε το αρχείο για ανάγνωση

	if(file == NULL) //αν δεν το ανοίξει τότε υπάρχει πρόβλημα
 	 { 

		printf("\n \n %s NOT FOUND \n \n",File);
		exit(1);

     }

    
	char number2 = ' '; // Αρχικοποιήση του number2 

	info.lines=0; //αρχικοποίηση αριθμού γραμμών
	info.cols=0; // αρχικοποίηση αριθμού στηλών
	int charas=0; //αρχικοποίηση αριθμού χαρακτήρων
	
	
	
	
	/*
	 * Μέτρημα χαρακτήρων για την εύρεση των γραμμών, στηλών και μεγέθους του πίνακα
	 * Δέχεται έναν χαρακτήρα και ελέγχει τι είναι:
	 *  αριθμός,κενό,[,], \n, not τελος αρχείου 
     */
	
	while (number2!= EOF)  //End Of File
		{
  			number2 = fgetc(file);//get the char from the file
   
   			if(number2 == '[' || number2 == ']') //αγκίλες
   			  {
      			 //skip this char
   			  }
   			else if (number2 == '\n')//change line
     		{
     			  //skip this char 
    
    			 printf("\n");// άλλαξε γραμμή για να μοιάζει με πίνακα ή αλλαγή γραμμής
     			 info.lines=info.lines+1; //αλλαγή γραμμής άρα γραμμή+1
   			  }
   			else if (number2 == ' ')
    		 {
     			   //skip this char
     			printf(" ");//καθαρά για την εμφάνιση του πίνακα
    		 }
   			else if(number2!=EOF)
  			   {  
   			    charas=charas+1; //μετρά τον αριθμό των χαρακτήρων
   			    putchar(number2);// εμφάνισε το 'γράμμα'
   			  }
      }

	//αριθμός στηλών
	info.cols = charas/info.lines; // (όλοι οι χαρακτήρες που δεν είναι ' ',\n,[,]) \ (αριθμός γραμμών)

	printf("the number of line is %d \n",info.lines); //counting how many change of lines were found 
	printf("the number of clumns is %d \n",info.cols); //number of columns


	fseek(file, 0, SEEK_SET);//reset the pointer to the beginning of the file

	char number3 = ' ';
	
	info.Size = info.cols*info.lines; //μέγεθος πίνακα
	
	int *FPinakas; // o 1D πίνακας που θα χρησιμοποιηθεί για τον gauss
	
	FPinakas = malloc(sizeof(int*) * info.Size);

	int i=0;
	int j=0;
	int kind;
	int num;

	/*
	 * Πέρασμα του πίνακα στον 1D Array FPinakas ανα γραμμή
	 */
	
	
	while (number3!= EOF)  //EOF = End Of File
		{
  			number3 = fgetc(file);// get the char
  			kind = 0;
		
		if(number3 == '[' || number3 == ']') //αγκίλες
    		 {
     		  //skip this char
     	     }
   		else if (number3 == '\n')//change line
             {
             //skip this char 
             }
       else if (number3 == ' ')
            {
            //skip this char   
            }
       else if(number3!=EOF)
            { 
              kind=1;
            }
	
	    
		
  	  if(kind==1)// αν χαρακτήρας
    		{   
  		       num = number3 - '0';
			   FPinakas[j] = num;
     	       //printf("pinakas[%d][%d] = %d \n", i,j,num);
    		   j=j+1; //next thesis
    		}
		
	   }
		



	fclose(file);//κλείσε το αρχείο

	
	return FPinakas;
	
}





//////2. Function For Writing The Array After Gauss To A File


/*Η συνάρτηση αυτή παίρνει σαν παράμετρο τον τελικό πίνακα
 * μετά από την εφαρμογή του Gauss και τον γράφει σε αρχείο.
 */


void WriteFile(int* data)
{

   FILE *final;
   final= fopen("final.txt","w");

   if(final == NULL)
   {
      printf("Error!");   
               
   }

   for(int a=0; a<info.lines;a++) // Για κάθε γραμμή
    {
	  fprintf(final, "["); //γράψε μία αγκύλη στην αρχή κάθε γραμμής
	  
	  for( int b=0;b<info.cols;b++) // Για κάθε στήλη
	     {
		   int c = b+a*info.cols; // Γράφει τους αριθμόυς ανά γραμμή 
		   fprintf(final, "%d", data[c]);
		 
		   //Για πίνακα χωρίς κενά μετά και πριν τις αγκίλες
		   if( b != info.cols-1)
		   	   {
			     fprintf(final, " ");
		   	   }

		   //printf("pin[%d][%d] = %d \n", a,b,pinakas[a][b]);
		 }
	   
	   fprintf(final, "]"); //αγκύλη στο τελος της γραμμής
	   fprintf(final, "\n"); //αλλαγή γραμμής
    }
	
	fclose(final);
	
}	


