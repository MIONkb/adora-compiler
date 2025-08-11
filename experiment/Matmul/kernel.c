// void vecadd(int a[20], int b[20], int c[20]){
//     int i;
//     #pragma scop
//     for(i = 0; i < 20; i++){
//         c[i] = a[i] + b[i];
//     }
//     #pragma endscop
//     return;
// }

#include "Matmul.h"


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


void merge_MATMUL_4x4(int a[W][N], int b[N][H], int c[W][H]) {
    #pragma scop
    for ( int i=0 ; i<W/4 ; i++ ) {
        for(int j=0 ; j<H/4 ; j++ ){
            c[4*i][4*j+0]   = 0;
            c[4*i][4*j+1]   = 0;
            c[4*i][4*j+2]   = 0;
            c[4*i][4*j+3]   = 0;
            c[4*i+1][4*j+0] = 0;
            c[4*i+1][4*j+1] = 0;
            c[4*i+1][4*j+2] = 0;
            c[4*i+1][4*j+3] = 0;
            c[4*i+2][4*j+0] = 0;
            c[4*i+2][4*j+1] = 0;
            c[4*i+2][4*j+2] = 0;
            c[4*i+2][4*j+3] = 0;
            c[4*i+3][4*j+0] = 0;
            c[4*i+3][4*j+1] = 0;
            c[4*i+3][4*j+2] = 0;
            c[4*i+3][4*j+3] = 0;
            for(int k=0 ; k<H ; ++k){
                c[4*i][4*j+0]   += a[4*i][k]   * b[k][4*j+0];
                c[4*i][4*j+1]   += a[4*i][k]   * b[k][4*j+1];
                c[4*i][4*j+2]   += a[4*i][k]   * b[k][4*j+2];
                c[4*i][4*j+3]   += a[4*i][k]   * b[k][4*j+3];
                c[4*i+1][4*j+0] += a[4*i+1][k] * b[k][4*j+0];
                c[4*i+1][4*j+1] += a[4*i+1][k] * b[k][4*j+1];
                c[4*i+1][4*j+2] += a[4*i+1][k] * b[k][4*j+2];
                c[4*i+1][4*j+3] += a[4*i+1][k] * b[k][4*j+3];
                c[4*i+2][4*j+0] += a[4*i+2][k] * b[k][4*j+0];
                c[4*i+2][4*j+1] += a[4*i+2][k] * b[k][4*j+1];
                c[4*i+2][4*j+2] += a[4*i+2][k] * b[k][4*j+2];
                c[4*i+2][4*j+3] += a[4*i+2][k] * b[k][4*j+3];
                c[4*i+3][4*j+0] += a[4*i+3][k] * b[k][4*j+0];
                c[4*i+3][4*j+1] += a[4*i+3][k] * b[k][4*j+1];
                c[4*i+3][4*j+2] += a[4*i+3][k] * b[k][4*j+2];
                c[4*i+3][4*j+3] += a[4*i+3][k] * b[k][4*j+3];
            }
        }
    }      
    #pragma endscop
}


void merge_MATMUL_4x4_affine(int a[W][N], int b[N][H], int c[W][H], int r[W][H]) {

    #pragma scop
    for ( int i=0 ; i<W ; ++i ) {
        for(int j=0 ; j<H ; j+=4 ){
            c[i][0]   = 0;
            c[i][1]   = 0;
            c[i][2]   = 0;
            c[i][3]   = 0;
            for(int k=0 ; k<N ; k++){
                c[i][0]   += a[i][k] * b[k][0];
                c[i][1]   += a[i][k] * b[k][1];
                c[i][2]   += a[i][k] * b[k][2];
                c[i][3]   += a[i][k] * b[k][3];
            }
        }
    }  
    // for (int i=0; i<W; ++i) {
    //     c[i][0] = 0;
    //     c[i][1] = 0;
    //     c[i][2] = 0;
    //     c[i][3] = 0;
    //     for (int k=0; k<N; ++k) {
    //         c[i][0] += a[i][k] * b[k][0];
    //         c[i][1] += a[i][k] * b[k][1];
    //         c[i][2] += a[i][k] * b[k][2];
    //         c[i][3] += a[i][k] * b[k][3];
    //     }
    // }

    #pragma endscop
}