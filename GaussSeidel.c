/*
 * GaussSeidel.c
 * G-S 迭代的串行实现.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_linalg.h>
#include <gsl_blas.h>
#include <gsl_eigen.h>
#include <gsl_complex_math.h>
#include "GaussSeidel.h"


#define USE_DRIVE 0             /* 是否编译 main 函数 */

#define TEST_GAUSSSEIDEL 1
#define DEBUG_RANDOM_MAT 0
#define DEBUG_MAT_GET 0
#define DEBUG_COND 0


#if USE_DRIVE

int
main(int argc, char *argv[]){

#if TEST_GAUSSSEIDEL

     int i,j;
     int size = SIZE;         /* 矩阵尺寸 */
     int cond_GS;
     double *A;            /* 系数矩阵 */
     double *x;            /* 方程组的解 */
     double *b;            /* 右端向量 */

     A = (double *)calloc(size*size, sizeof(double));
     b = (double *)calloc(size, sizeof(double));
     x = (double *)calloc(size, sizeof(double));

     /* 随机生成矩阵, 直到满足条件 */
     do
     {
          mat_get(A, size);
          mat_cond_GS(A, size, &cond_GS);
     } while (cond_GS != 1);
     /* 生成右端向量 */
     vec_get(b, size);
#if OUTPUT
     printf("Matrix generate success.\n");
     printf("A is :\n");
     for (i = 0; i < size; ++i){
          for (j = 0; j < size; ++j)
               printf("%7.2f", mat(A,size,i,j));
          printf("\n");
     }
     printf("b is :\n");
     for (i = 0; i < size; ++i)
          printf("%7.2f\n", b[i]);

#endif /* OUTPUT */

     gsl_vector_view x_view = gsl_vector_view_array(x, size);
     gsl_vector_set_all (&x_view.vector, 1.0); /* first guess */
     GaussSeidel(A, b, x, size);


#endif /* TEST_GAUSSSEIDEL */


#if OUTPUT
     printf("x is :\n");
     for (i = 0; i < size; ++i)
          printf("%7.2f\n", x[i]);

#endif /* OUTPUT */

     check_result(A, b, x, size);

     free(b);
     free(x);
     free(A);

#if DEBUG_RANDOM_MAT

     int i,j,count;
     int size = SIZE;
     int cond_GS;
     double *A;

     A = (double *)calloc(size*size, sizeof(double));

     /* 随机生成矩阵, 直到满足条件 */
     count = 0;
     do
     {
          mat_get(A, size);
          mat_cond_GS(A, size, &cond_GS);
          count++;
          printf("count = %d\n", count);
     } while (cond_GS != 1);

     printf("Count is : %d\n", count);
     free(A);

#endif /* DEBUG_RANDOM_MAT */

#if DEBUG_COND
     int size = 3;
     int cond_GS;
     double a_data[] = { 1, 1, 2,
                         -1, 1, 0,
                         -1, 0, 1
     };

     mat_cond_GS(a_data, size, &cond_GS);
     printf("cond_GS  = %d\n", cond_GS);

#endif /* DEBUG_COND */

#if DEBUG_MAT_GET
     int i,j;
     double *A;
     int size = SIZE;

     A = (double *)calloc(size*size, sizeof(double));
     mat_get(A, size);
     for (i = 0; i < size; ++i){
          for (j = 0; j < size; ++j){
               printf("%7.2f", mat(A,size,i,j));
          }
          printf("\n");
     }
     free(A);
#endif /* DEBUG_MAT_GET */

     return 0;
}


#endif /* USE_DRIVE */



/*---------------------------------------------------------------------
 * Function:  GaussSeidel
 * Purpose:   G-S 迭代的串行实现
 *            1, 收敛条件设置为 norm2(x^(n+1)-x^n)/norm2(x^{n+1}) < e
 *            2, 迭代矩阵用 Jacobi 迭代矩阵, M_J = D^{-1} (U+L)
 *               如果沿标号顺序生成 x^{n+1} , 那么这与使用 GaussSeidel
 *               迭代矩阵是等价的, 并且省去了求三角矩阵逆以及一次矩阵相乘的运算.
 *               其中 : A = D-U-L
 * In args:
 *    A         --系数矩阵
 *    b         --右端向量
 *    size      --矩阵规模
 * Out args:
 *    x         --方程组的解
 */
void
GaussSeidel(double *A, double *B, double *X, int size){
     int i,j;
     double norm2;         /* 向量范数 */
     double det_norm2;     /* 前后两次结果之差的范数. */
     double *M;            /* 迭代矩阵 M = D^{-1}(U+L) */
     double *G;            /* g = (D-L)^{-1}*b */
     int count;            /* 迭代的次数 */
     M = (double *)calloc(size*size, sizeof(double));
     G = (double *)calloc(size, sizeof(double));

     gsl_vector_view x = gsl_vector_view_array(X, size);
     gsl_matrix_view a = gsl_matrix_view_array(A, size, size);
     gsl_matrix_view m = gsl_matrix_view_array(M, size, size);
     gsl_vector_view b = gsl_vector_view_array(B, size);
     gsl_vector_view g = gsl_vector_view_array(G, size);

     GaussSeidel_pre(A, B, M, G, size);

#if OUTPUT
     printf("M is :\n");
     for (i = 0; i < size; ++i){
          for (j = 0; j < size; ++j)
               printf("%7.2f", mat(M,size,i,j));
          printf("\n");
     }

     printf("g is :\n");
     for (i = 0; i < size; ++i)
          printf("%7.2f\n", G[i]);

#endif /* OUTPUT */

/*--------------------------G-S 迭代----------------------------------*/

     count =0;
     double xi;            /* 新 x_i */
     double xi_pre;        /* 旧 x_i */

     do
     {
          det_norm2 = 0.0; /* 这个值累加 */
          for (i = 0; i < size; ++i){
               xi_pre = gsl_vector_get(&x.vector, i); /* 旧的 x_i */

               /* 计算新的 x_i  */
               gsl_vector_view row = gsl_matrix_row(&m.matrix, i);
               gsl_blas_ddot (&row.vector, &x.vector, &xi); /* 计算内积 */
               xi += gsl_vector_get(&g.vector, i);          /* +g */
               gsl_vector_set (&x.vector, i, xi);

               /* 计算 x_n-x_{n+1} 的范数 */
               det_norm2 += (xi-xi_pre)*(xi-xi_pre);
          }
          det_norm2 = sqrt(det_norm2);
          norm2 = gsl_blas_dnrm2 (&x.vector); /* 求二范数 */

          count++;
     } while ( det_norm2/norm2 > TOL);
#if OUT_COUNT
     printf("GaussSeidel, 迭代次数 %d\n", count);
#endif /* OUT_COUNT */


     return;
}


/*---------------------------------------------------------------------
 * Function:  mat_get
 * Purpose:   生成随机矩阵
 * In args:
 *    size      --矩阵的规模
 * Out args:
 *    A         --矩阵指针
 */
void
mat_get(double *A, int size){

#if TEST_MAT
     int i,j;
     int msize = 5;
     double B[] = {5.00 ,   0.17,   3.71,   0.33,   0.40,
                   -2.94        ,   8.00,  -1.89,   4.58,  -0.89,
                   -3.26        ,  -1.77,   7.00,   1.51,  -1.66,
                   1.89 ,   4.89,  -0.67,   8.00,  -2.77,
                   1.23 ,   4.40,  -0.64,   1.81,  10.00,
     };
     for (i = 0; i < msize*msize; ++i){
          A[i] = B[i];
     }

#else
     int i,j;
     double temp;

     /* 生成随即矩阵 */
     vec_get(A, size*size);

     /* 使对角元素严格占优 */
     for (i = 0; i < size; ++i){
          temp = 0;
          for (j = 0; j < size; ++j){
               temp+=abs(mat(A,size,i,j));
          }
          mat(A,size,i,i)=temp*PARTIAL;
     }

#endif /* TEST_MAT */

     return;
}

/*---------------------------------------------------------------------
 * Function:  vec_get
 * Purpose:   生成随机向量
 * In args:
 *    size      --向量长度
 * Out args:
 *    v         --生成的向量
 */
void
vec_get(double *v, int size){
#if TEST_MAT
     int i;
     int msize = 5;
     double V[]= {9.61,
                  6.86,
                  1.82,
                  11.34,
                  16.80
     };
     for (i = 0; i < msize; ++i){
          v[i] = V[i];
     }
#else
     int i;
     double range = RANGE;

     srandom(time(NULL));
     for(i=0;i<size;i++)
          *(v+i)=-0.5*range+range*(random() / ((double)RAND_MAX+1.0));
#endif /* TEST_MAT */

     return;
}


/*---------------------------------------------------------------------
 * Function:  mat_cond_GS
 * Purpose:   判断矩阵是否足够非奇异, 以及是否满足 G-S 迭代收敛的条件
 *            1. 矩阵的条件数 A_cond = ||A^{-1}|| * ||A||, 表征了奇异程度
 *               这里使用 l^2 范数, 那么 A_cond = S_n/S_1 即最大最小奇异值之比.
 *            2. G-S 迭代收敛的充要条件: rho(M_g)<1
 *               rho 是谱半径, M_g = (D-L)^{-1} U 是迭代矩阵. A = D-L-U
 *               将上界设为 rho_up<<1 可以有较快的收敛速度.
 * In args:
 *    A         --要判断的矩阵
 *    size      --矩阵规模
 * Out args:
 *    cond_GS   --是否满足条件
 */
void
mat_cond_GS(double *A, int size, int *cond_GS){
     double rho_up  = 0.8;     /* 谱半径上限 */
     double cond_up = 10;      /* 条件数上限 */

     int i, j;
     double cond_A;           /* A 的条件数 */
     double rho_M;            /* M_g 的谱半径 */
     double *A_dup;           /* dup of A */
     double *M;               /* G-S 迭代矩阵 */

     A_dup      = (double *)calloc(size*size, sizeof(double));
     M          = (double *)calloc(size*size, sizeof(double));

     /* 让 gsl 知道怎么访问连续的存储空间 */
     gsl_matrix_view a          = gsl_matrix_view_array (A, size, size);
     gsl_matrix_view a_dup      = gsl_matrix_view_array (A_dup, size, size);
     gsl_matrix_view m          = gsl_matrix_view_array (M, size, size);

     gsl_matrix_memcpy (&a_dup.matrix, &a.matrix); /* 将A的值复制到A_dup */
     gsl_matrix_memcpy (&m.matrix, &a.matrix);     /* 将A的值复制到M */

     /** 计算 A 的条件数, 用 svn 分解计算奇异值 **/
     gsl_matrix *V = gsl_matrix_alloc(size, size);   /* svn V */
     gsl_vector *w_svn = gsl_vector_alloc(size);     /* svn workspace */
     gsl_vector *sv = gsl_vector_alloc(size);        /* singular values */

     gsl_linalg_SV_decomp (&a_dup.matrix, V, sv, w_svn);
     cond_A = gsl_vector_get(sv, 0)/gsl_vector_get(sv, size-1);

     /** 计算 M 的谱半径 **/
     gsl_matrix *I = gsl_matrix_alloc (size, size); /* 单位矩阵 */
     gsl_matrix_set_identity(I);
     gsl_eigen_nonsymm_workspace * w_eigen =
          gsl_eigen_nonsymm_alloc(size); /* eigen work space */
     gsl_vector_complex *eval = gsl_vector_complex_alloc(size); /* 特征值 */

     /* 计算矩阵 M_g , M_g = I - (D-L)^{-1}*A */
     /* B = a*inv(A)*B, A 是三角矩阵 */
     gsl_blas_dtrsm(CblasLeft,    /* inv(A)*B */
                    CblasLower,   /* 取 A 的下三角部分 */
                    CblasNoTrans, /* 不转至 */
                    CblasNonUnit, /* A 的对角线不是单位的 */
                    1.0,          /* a */
                    &a.matrix, &m.matrix
          );                              /* 算(D-L)^{-1} * A */
     gsl_matrix_scale(&m.matrix, -1.0);   /* *(-1) */
     gsl_matrix_add(&m.matrix, I);        /* +I */

     /* 计算 M_g 的特征值. */
     gsl_complex eval_i;
     double evaln_i;                                /* 第 i 个特征值的范数 */
     gsl_eigen_nonsymm_params (0, 0, w_eigen);      /* 设定 w 的参数 */
     gsl_eigen_nonsymm (&m.matrix, eval, w_eigen);  /* 求特征值 */

     /* 算谱半径 */
     rho_M = 0.0;
     for (i = 0; i < size; ++i){
          eval_i  = gsl_vector_complex_get(eval, i);
          evaln_i = gsl_complex_abs(eval_i);
          if (rho_M < evaln_i)
               rho_M = evaln_i;
     }

     /** 判断是否满足条件 **/
     *cond_GS = (cond_A < cond_up && rho_M < rho_up) ? 1 : 0;

#if  OUTPUT

     printf("cond_A     = %10.3e\n", cond_A);
     printf("rho_M      = %10.3e\n", rho_M);

#endif /* OUTPUT */

     free(A_dup);
     free(M);
     gsl_vector_free(w_svn);
     gsl_vector_free(sv);
     gsl_matrix_free(V);
     gsl_matrix_free(I);
     gsl_vector_complex_free(eval);
     gsl_eigen_nonsymm_free (w_eigen);
     return;
}

/*---------------------------------------------------------------------
 * Function:  check_result
 * Purpose:   检验求解的结果, x 是之前的计算值
 *            只需检验是否成立 A x = b
 * In args:
 *   A          --系数矩阵
 *   b          --右端向量
 *   x          --要检验的解
 *   size       --矩阵规模
 */
void
check_result(double *A, double *B, double *X, int size){
     double errnorm2;      /* 误差的二范数 */

     double *B_dup = (double *)calloc(size, sizeof(double));

     /* 告诉 gsl 怎么读数据 */
     gsl_matrix_view a          = gsl_matrix_view_array (A, size, size);
     gsl_vector_view b          = gsl_vector_view_array (B, size);
     gsl_vector_view x          = gsl_vector_view_array (X, size);
     gsl_vector_view b_dup      = gsl_vector_view_array (B_dup, size);

     /* 计算 Ax = b_dup */
     gsl_blas_dgemv (CblasNoTrans,
                     1.0,
                     &a.matrix,
                     &x.vector,
                     0.0,
                     &b_dup.vector
          );

     /* 计算误差 */
     gsl_vector_sub (&b_dup.vector, &b.vector);
     errnorm2 = gsl_blas_dnrm2 (&b_dup.vector); /* 求误差的二范数 */

     /* 判断是否是准确解 */
     if( errnorm2 < EPS )
          printf("GaussSeidel success!\n");
     else
          printf("GaussSeidel failed. errnorm2 = %e\n", errnorm2);

     free(B_dup);
     return;
}

/*---------------------------------------------------------------------
 * Function:  GaussSeidel_pre
 * Purpose:   生成 G-S 迭代需要的迭代矩阵 M_J 和向量 g
 *            迭代矩阵 M = D^{-1}(U+L)
 *            g = (D-L)^{-1}*b
 * In args:
 *    A         --系数矩阵
 *    B         --右端向量
 *    size      --矩阵大小
 * Out args:
 *    M         --迭代矩阵
 *    G         --迭代向量
 */
void
GaussSeidel_pre(double *A, double *B, double *M, double *G, int size){
     int i;
     double *diag;         /* M 的对角线 D */

     diag = (double *)calloc(size, sizeof(double));

     /* 让 gsl 知道怎么读矩阵和向量 */
     gsl_matrix_view a = gsl_matrix_view_array(A, size, size);
     gsl_matrix_view m = gsl_matrix_view_array(M, size, size);
     gsl_vector_view b = gsl_vector_view_array(B, size);
     gsl_vector_view g = gsl_vector_view_array(G, size);

     gsl_vector_view d = gsl_matrix_diagonal(&a.matrix); /* D, A对角线 */
     /** 求 M 和 G **/
     gsl_matrix_memcpy (&m.matrix, &a.matrix); /* 将A的值复制到M */
     gsl_vector_memcpy (&g.vector, &b.vector); /* 将B的值附到G */

     /* 定义单位矩阵 */
     gsl_matrix *I = gsl_matrix_alloc (size, size);
     gsl_matrix_set_identity(I);

     /* 计算 g = D^{-1}b*/
     gsl_vector_div(&g.vector, &d.vector); /* a = a/b */

     /* 计算 M = D^{-1}(L+U) = -D^{-1}A + I */
     for (i = 0; i < size; ++i){
          gsl_vector_view row = gsl_matrix_row(&m.matrix, i); /* 考察一行 */
          gsl_vector_scale (&row.vector,            /* 整行乘标量 */
                            1.0/gsl_vector_get(&d.vector, i));
     }
     gsl_matrix_scale(&m.matrix, -1.0);   /* *(-1) */
     gsl_matrix_add(&m.matrix, I);        /* +I */

     return;
}
