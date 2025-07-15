// void vecadd(int a[20], int b[20], int c[20]){
//     int i;
//     #pragma scop
//     for(i = 0; i < 20; i++){
//         c[i] = a[i] + b[i];
//     }
//     #pragma endscop
//     return;
// }


#define size 4


//kernel 23
void kernel_merge4(int a[size], int b[size], int c[size], int d[size], int r[4*size]) {
    #pragma scop
    for ( int i=0 ; i<size ; i++ ) {
        for(int j=0 ; j<4 ; j++){
            r[4* i] = a[i];
            r[4* i + 1] = b[i];
            r[4* i + 2] = c[i];
            r[4* i + 3] = d[i];
        }
    }      
    #pragma endscop
}
