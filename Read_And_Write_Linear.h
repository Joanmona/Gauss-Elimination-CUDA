//Header File Of Read_And_Write_Linear.c


struct ArrayInfo{

	int lines; //αριθμός γραμμών
	int cols; //αριθμός στηλών
	int Size;  //μέγεθος πίνακα

};

int* ReadFile(char*); //Διαβάζει το αρχείο και περνάει τον αρχικό πίνακα σε 1D Array
void WriteFile(int*); //Περνάει τον τελικό πίνακα από 1D Array σε αρχείο

extern struct ArrayInfo info; // var για να χρησιμοποιειται και απο τα δύο αρχεία