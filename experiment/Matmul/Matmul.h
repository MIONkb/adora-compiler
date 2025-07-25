#define W 4
#define N 4
#define H 4

#define Dtype int

void unroll_MATMUL(Dtype a[W][N], Dtype b[N][H], Dtype c[W][H], Dtype r[W][H]) ;
void merge_MATMUL(Dtype a[W][N], Dtype b[N][H], Dtype c[W][H], Dtype r[W][H]) ;