/* mpiGaussSeidel.c
 * 采用流水线的思想设计并行高斯 -赛德尔迭代算法
 * 需要使用 GaussSeidel.c 中的函数, 因此要包含头文件 GaussSeidel.h
 *
 * Fri May 16 23:35:15 2014
 * Author: Xu KaiWen <xukaiwen@lsec.cc.ac.cn>
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <gsl_blas.h>
#include "GaussSeidel.h"
#include "timer.h"

int
main(int argc, char *argv[]){
   int i,j;
   int size = N;         /* 矩阵尺寸 */
   int cond_GS;
   int myrank;
   int count;            /* 记录尝试获取矩阵的次数 */
   int error;
   double *A;            /* 系数矩阵 */
   double *x;            /* 方程组的解 */
   double *b;            /* 右端向量 */

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   if (myrank == 0){
      A = (double *)calloc(size*size, sizeof(double));
      b = (double *)calloc(size, sizeof(double));
   }
   x = (double *)calloc(size, sizeof(double));

   if (myrank == 0){
      /* 随机生成矩阵, 直到满足条件 */
      count = 0;
      do
      {
         mat_get(A, size);
         mat_cond_GS(A, size, &cond_GS);
         count++;
      } while (cond_GS != 1);
      /* 生成右端向量 */
      vec_get(b, size);

      printf("Number of atempts to get matrix: %d\n", count);
   }
#if OUT_MAT
   /* 打印出生成的矩阵 */
   if (myrank == 0){
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
   }
#endif /* OUT_MAT */

   /* First guess. */
   gsl_vector_view x_view = gsl_vector_view_array(x, size);
   gsl_vector_set_all(&x_view.vector, 1.0); /* 将元素初始化为 0 */

   MPI_Barrier(MPI_COMM_WORLD);

#if USEMPI
   mpiGaussSeidel(A, b, x, size);
#else
   if (myrank == 0){
      GaussSeidel(A, b, x, size);
   }
#endif /* USEMPI */


#if OUT_X
   if(myrank == 0){
      printf("x is :\n");
      for (i = 0; i < size; ++i)
         printf("%7.2f\n", x[i]);
   }
#endif /* OUT_X */

/* 检测运算结果 */
   if (myrank == 0){
      check_result(A, b, x, size);
   }

   fflush(stdout);
   MPI_Barrier(MPI_COMM_WORLD);
   printf("proc %d, final\n", myrank);
#if DEBUG

   char message[MPI_MAX_ERROR_STRING];
   int meslen;
   error = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

#endif /* DEBUG */

   /* error = MPI_Finalize(); */
#if DEBUG

   if ( error != MPI_SUCCESS){
      MPI_Error_string(error, message, &meslen);
      printf("line %d, error code is %d, message: %s\n",
             __LINE__, error, message);
      MPI_Abort(MPI_COMM_WORLD, 1);
   }

#endif /* DEBUG */

   if (myrank == 0){
      free(b);
      free(A);
   }
   free(x);
   return 0;
}

/*---------------------------------------------------------------------
 * Function:  mpiGaussSeidel
 * Purpose:   用 mpi 实现流水线并行 GaussSeidel 算法
 *            采用矩阵按行的卷帘存储, 见 bxjs --zlb et al. pp. 277
 * In args:
 *    A         --系数矩阵
 *    b         --右端向量
 *    size      --矩阵规模
 * Out args:
 *    x         --方程组的解
 */
void
mpiGaussSeidel(double *A, double *b, double *x, int size){

   int error;
#if DEBUG

   char message[MPI_MAX_ERROR_STRING];
   int meslen;
   error = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

#endif /* DEBUG */

   int i,j,k;
   int myrank, np;       /* 进程号, 进程数 */
   int l;                /* l = size%np 不能统一分配部分的大小 */
   int n_loc;            /* 该进程存储的行数 */
   double *M_loc;         /* 进程存储的矩阵部分 */
   int root = 0;         /* 根进程 */
   double *g;            /* g = (D-L)^{-1}*b */
   g = (double *)calloc(size, sizeof(double));

   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   MPI_Request *sreq = (MPI_Request *)calloc(np, sizeof(MPI_Request));
   MPI_Request *rreq = (MPI_Request *)calloc(np, sizeof(MPI_Request));
   /* 这样定义的 req 在用 wait 之后是不需要再释放的 */
   MPI_Status  *sta = (MPI_Status  *)calloc(np, sizeof(MPI_Status));
   /* 初始化 */
   for (i = 0; i < np; ++i){
      sreq[i] = MPI_REQUEST_NULL;
      rreq[i] = MPI_REQUEST_NULL;
   }

   l = size%np;
   n_loc = size/np + ((myrank < l )? 1 : 0); /* M_loc 的行数 */
   M_loc = (double *)calloc(size*n_loc, sizeof(double));

   MPI_Datatype row;                 /* 一行 */
   MPI_Datatype T_loc;               /* 该进程要存储的矩阵数据类型 */
   error = MPI_Type_contiguous(size, MPI_DOUBLE, &row);
   error = MPI_Type_contiguous(n_loc, row, &T_loc);
   error = MPI_Type_commit(&row);
   error = MPI_Type_commit(&T_loc);

   if (myrank == root){
      double norm2;         /* 向量范数 */
      double det_norm2;     /* 前后两次结果之差的范数. */
      double *M;            /* 迭代矩阵 M = D^{-1}(U+L) */

      M = (double *)calloc(size*size, sizeof(double));

      GaussSeidel_pre(A, b, M, g, size); /* 计算 M 和 g */

#if OUT_MLOC
      /* 输出迭代矩阵 M */
      if (myrank == 0){
         printf("M is :\n");
         for (i = 0; i < size; ++i){
            for (j = 0; j < size; ++j)
               printf("%7.2f", mat(M,size,i,j));
            printf("\n");
         }
         printf("g is :\n");
         for (i = 0; i < size; ++i){
            printf("%7.2f\n", g[i]);
         }
      }
#endif /* OUT_MLOC */
      /* 按卷帘存储方式分发矩阵 */
      MPI_Datatype T_locl;  /* size/np + 1 */
      MPI_Datatype T_locs;  /* size/np */
      error = MPI_Type_vector(size/np+1, 1, np, row, &T_locl);
      error = MPI_Type_vector(size/np, 1, np, row, &T_locs);

      int ext_row;
      int lb_row;
      error = MPI_Type_get_extent(row, &lb_row, &ext_row);
      /* printf("ext_row = %d\n", ext_row); */
      /* MPI_Type_create_resized(T_locl, 0, ext_row, &T_locl); */
      /* MPI_Type_create_resized(T_locs, 0, ext_row, &T_locs); */
      error = MPI_Type_commit(&T_locs);
      error = MPI_Type_commit(&T_locl);

      for (i = 0; i < np; ++i){ /* 前 l 个进程的行数要 +1 */
         if (i<l){
            error = MPI_Isend(M+i*size, 1, T_locl,
                              i, 0, MPI_COMM_WORLD, rreq+i);

         } else {
            error = MPI_Isend(M+i*size, 1, T_locs,
                              i, 0, MPI_COMM_WORLD, rreq+i);
         }
      }
   }
   MPI_Status st;
   MPI_Request req;

   /* 接受局部矩阵 */
   error = MPI_Irecv(M_loc, 1, T_loc, root, 0, MPI_COMM_WORLD, &req);
   /* 分发向量 g */
   error = MPI_Bcast(g, 1, row, root,  MPI_COMM_WORLD);

#if OUT_MLOC
   sleep(myrank);
   /* 输出局部矩阵 */
   printf("M_loc of proc %d\n", myrank);
   for (i = 0; i < n_loc; ++i){
      for (j = 0; j < size; ++j){
         printf("%7.2f", mat(M_loc,size,i,j));
      }
      printf("\n");
   }
#endif /* OUT_MLOC */

   /* 等待所有通讯完成. */
   if (myrank == root){
      error = MPI_Waitall(np, rreq, sta);
   }
   error = MPI_Wait(&req, &st);


/*--------------------------并行 G-S 迭代-----------------------------*/

   int count = 0;        /* 计数器, 迭代次数 */
   int own;              /* 单元的属主进程. */
   int i_glb;            /* 全局指标 */
   int start;            /* x 开始的指标 */
   int end;		 /* 循环结束的指标 */
   int *map;		 /* 处理器编号的映射，采用局部的视角
			  * 这样在写指标的时候，就可以避免出现 myrank */
   double xi;            /* 新 x_i, 为了算误差范数 */
   double xi_pre;        /* 旧 x_i */
   double err_norm2;     /* 相邻迭代结果之差的二范数 */
   double norm2;         /* 迭代向量的二范数 */
   gsl_vector_view x_view = gsl_vector_view_array(x, size);
   map = (int *)calloc(np-1, sizeof(int));
   for (i = 1; i < np; ++i){
      map[i]=(myrank + i)%np;
   }
#if DEBUG
   printf("Before mpiGS\n");
#endif /* DEBUG */

#if TEST_FIRST
   /* 一次迭代 */
   do
   {
#endif /* TEST_FIRST */
      err_norm2 = 0.0;
      /* 计算一行 */
      for (i = 0; i < n_loc; ++i){
         i_glb = i*np + myrank; /* 要更新的元素的全局位置 */
         xi_pre = x[i_glb];
         xi = 0.0;            /* 这个值在计算迭代过程中累加 */
         /* 非阻塞的接收, 以实现计算与通讯的重叠 */
         start = i_glb-np+1;                   /* 此时对i_glb_np
                                                * 的写入是非法的 */
         end = i_glb;
         for (k = start; k < end; ++k){ /* k 也是全局指标, */
            if ( !(k<0) ){
               own = k%np; /* owner */
               error = MPI_Irecv(x+k, 1, MPI_DOUBLE,
                                 own, k, /* tag 的值与元素所在的位置相同. */
                                 MPI_COMM_WORLD,
                                 rreq+k-start+1 /* rreq 从 0 号位置开始编号 */
                  );
            }
         }

         /* 从这里开始计算新的 xi */
         /* 因为在算上一行的时候已经用过这部分的元素, 所以是最新的. */
         start = i_glb;
         end = i_glb+size-np+1; /* 最后 np-1 个元素是要等待更新的 */
         for (k=start; k<end; ++k){
            xi += mat(M_loc,size,i,k%size)*x[k%size];
         }
         /* 这是需要等待其他进程更新的部分 */
         start = i_glb+size-np+1;
         end = i_glb+size;
         for (k = start; k <end ; ++k){
            error = MPI_Wait(rreq+k-start+1, sta+k-start+1);
            xi += mat(M_loc,size,i,k%size)*x[k%size];
         }
         xi+=g[i_glb];
         /* x_i 在此处已经算好 */
         x[i_glb]=xi;

         /* 先检测之前的 send 是否做完 */
         error = MPI_Waitall(np, sreq, sta);
         /* 将 x_i 分发到所有进程 */
         for (k = 0; k < np; ++k)
            if (k != myrank) /* 不需要对自己发数据 */
               error = MPI_Isend(x+i_glb, 1, MPI_DOUBLE,
                                 k, i_glb, MPI_COMM_WORLD, sreq+k);
         /* 相邻步之差的二范数 */
         err_norm2 += (xi-xi_pre)*(xi-xi_pre);
      }
      /* 完成消息传递, 要做这一步才能使所有进程中的 x 都得到更新 */
      double offset = (myrank-l+1+np)%np; /* 到最后一个进程的偏移量 */
      if (myrank == l-1)                  /* 第 l-1 号进程是最后一
                                           * 个发数据的进程 */
         offset = np;
      start = size-np+offset;           /* np-offset 是需要做的循环的长度 */
      end = size;
      for (i = start; i < size; ++i){
         own = i%np;
         error = MPI_Irecv(x+i, 1, MPI_DOUBLE, own, i,
                           MPI_COMM_WORLD,
                           rreq+(myrank-l+np)%np-1+i-start);
      } /* 完成一次迭代 */

      error = MPI_Waitall(np, sreq, sta);
      error = MPI_Waitall(np, rreq, sta);

/* 将所有的 err_norm2 求和 */
      error = MPI_Allreduce(&err_norm2, &norm2, /* norm2 做为临时存储空间 */
                            1, MPI_DOUBLE,
                            MPI_SUM, MPI_COMM_WORLD);
      err_norm2 = sqrt(norm2);
      norm2 = gsl_blas_dnrm2 (&x_view.vector); /* 求二范数 */

      count++;
#if DEBUG

      printf("Proc %d, end of the %d loop\n", myrank, count);

#endif /* DEBUG */
#if TEST_FIRST
   } while ( err_norm2/norm2 > TOL);
#endif /* TEST_FIRST */

#if OUT_COUNT
   if (myrank == root){
      printf("mpiGaussSeidel, proc %d, 迭代次数%d \n",
             myrank, count);
   }
#endif /* OUT_COUNT */

   if (myrank == root){
      free(M_loc);
   }
   free(g);
   for (i = 0; i < np; ++i){
      if (*(sreq+i) != MPI_REQUEST_NULL) /* NULL的是已经回收过内存的 */
         error = MPI_Request_free(sreq+i);
      if (*(rreq+i) != MPI_REQUEST_NULL)
         error = MPI_Request_free(rreq+i);
   }
   free(sta);

   return;
}
