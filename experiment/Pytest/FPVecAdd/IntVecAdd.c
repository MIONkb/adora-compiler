void IntVecAdd(int a[20], int b[20], int c[20])
{
    #pragma scop
    for(int i = 0; i < 20; i++){
        c[i] = a[i] + b[i];
    }
    #pragma endscop    
}