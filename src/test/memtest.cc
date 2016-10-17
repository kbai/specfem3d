#include <stdio.h>
#include <stdlib.h>
int main()

{
	float* a = new float[1000000];
//	float b[1000000];
//	b[999999]=1.0;
	a[999999] = 1.0;
	for(int i = 0;i < 1000000; i++)
	{
		for(int j = 0; j < 1 ; j++) 
			a[i] = i*1.0;
	}
	printf("\nvalue of the array: %f\n",a[999999]);
	delete[] a;
}
