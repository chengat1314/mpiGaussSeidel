/*
 * GaussSeidel.c 的头文件
 * 是为了在 mpiGaussSeidel.c 中调用下面这些函数
 * 这样使用时, 需要将 GaussSeidel.c 中的 USE_DRIVE 设置为 0
 */

#ifndef _GAUSSSEIDEL_H_
#define _GAUSSSEIDEL_H_

#define RANGE 10.0                    /* 随机数的取值范围 */
#define PARTIAL 0.5                   /* 生成矩阵对角元与行和的比值 */
#define TOL 1.0e-5                    /* 迭代终止的判别条件. */
#define EPS 1.0e-3                    /* 解的允许误差 */
#define mat(M,len,i,j) (*(M+i*len+j)) /* 将一维数组读为矩阵 */

/* 这些参数决定了需要编译的部分 */
#define USEMPI 1		/* 是否使用 mpi 版本的 GS迭代 */
#define OUT_MAT 0		/* 是否打印出生成的矩阵和向量 */
#define OUT_X 0			/* 是否打印出结果 */
#define OUT_MLOC 0		/* 输出迭代矩阵 M 以及局部矩阵 M_loc */
#define OUT_COUNT 1		/* 是否打印出循环次数 */
#define TEST_FIRST 1		/* 查看一次迭代的结果, 设置为 0 只迭代一次 */
#define TEST_MAT 0		/* 使用实现给定的矩阵和右端向量 */
#define DEBUG 0

#define SIZE 100			/* 矩阵规模 */

#if TEST_MAT
#define SIZE 5                           /* 测试矩阵规模 */
#endif /* TEST_MAT */



void mat_get(double *A, int size);
void vec_get(double *v, int size);
void mat_cond_GS(double *A, int size, int *cond_GS);
void GaussSeidel(double *A, double *b, double *x, int size);
void check_result(double *A, double *b, double *x, int size);
void GaussSeidel_pre(double *A, double *B, double *M, double *G, int size);
void mpiGaussSeidel(double *A, double *b, double *x, int size);

#endif /* _GAUSSSEIDEL_H_ */
