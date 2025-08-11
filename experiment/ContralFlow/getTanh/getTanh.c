#include <stdlib.h>

int getTanh(int A[100]){

  int i,beta;
  int result ;
  int d = 0;
  for (i = 0; i < 100; i++){
       beta = A[i];

		if (beta < 1){
          result = ((beta*beta+19)*beta*beta+3)*beta;
        }
        else{

          result = 1.0;
        }

        d += result;
  }
  return d;

}

