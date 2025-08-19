#define W 36
#define N 36
#define H 36

#define Dtype int

void unroll_MATMUL(Dtype a[W][N], Dtype b[N][H], Dtype c[W][H], Dtype r[W][H]) ;
void merge_MATMUL(Dtype a[W][N], Dtype b[N][H], Dtype c[W][H], Dtype r[W][H]) ;