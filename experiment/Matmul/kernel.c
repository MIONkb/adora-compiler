// void vecadd(int a[20], int b[20], int c[20]){
//     int i;
//     #pragma scop
//     for(i = 0; i < 20; i++){
//         c[i] = a[i] + b[i];
//     }
//     #pragma endscop
//     return;
// }


#define W 4
#define N 4
#define H 4


void unroll_MATMUL(int a[W][N], int b[N][H], int c[W][H], int r[W][H]) {
    #pragma scop
    for ( int i=0 ; i<W ; i++ ) {
        for(int j=0 ; j<H ; j++){
            r[i][j] = c[i][j];
            for(int k=0 ; k<W ; k++){
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }      
    #pragma endscop
}

void merge_MATMUL(int a[W][N], int b[N][H], int c[W][H], int r[W][H]) {
    #pragma scop
    for ( int i=0 ; i<W ; i++ ) {
        r[i][0] = c[i][0];
        r[i][1] = c[i][1];
        r[i][2] = c[i][2];
        r[i][3] = c[i][3];
        for(int k=0 ; k<W ; k++){
            r[i][0] += a[i][k] * b[k][0];
            r[i][1] += a[i][k] * b[k][1];
            r[i][2] += a[i][k] * b[k][2];
            r[i][3] += a[i][k] * b[k][3];
        }
    }      
    #pragma endscop
}
