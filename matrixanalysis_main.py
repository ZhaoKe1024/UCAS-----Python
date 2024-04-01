# -*- coding: utf-8 -*-
# @Author : Z *, id:2022********026
# @Time : 2022-11-13 20:57
import numpy as np

"""
题目
要求完成课堂上讲的关于矩阵分解的LU、QR（Gram-Schmidt）、Orthogonal Reduction (Householder reduction
    和Givens reduction)和 URV程序实现，要求如下：
1、一个综合程序，根据选择参数的不同，实现不同的矩阵分解；在此基础上，实现Ax=b方程组的求解，以及计算A的行列式；
2、可以用matlab、Python等编写程序，需附上简单的程序说明，比如参数代表什么意思，输入什么，输出什么等等，附上相应的例子；
3、一定是可执行文件，例如 .m文件等,不能是word或者txt文档。附上源代码，不能为直接调用matlab等函数库;
"""

"""
README
学号: 2022********026
K Z

有以下13个函数可以调用
0. get_matrix: 返回用于测试案例的矩阵
1. gauss_jordan: 输入一个任意形状的矩阵，输出其行阶梯形矩阵；采用Gauss-Joran法，类似于部分主元法
    用于求解线性方程组和行列式
2. luf： 输入一个方阵（必须方阵），输出矩阵 L、U、P；即PA=LU分解

# QR分解和URV分解，不限制矩阵形状
3. qr_gram_schmidt：输入一个任意形状的矩阵，输出矩阵Q、R
4. qr_householder：输入一个任意形状的矩阵，输出矩阵Q、R
5. qr_givens：输入一个任意形状的矩阵，输出矩阵Q、R
6. urv：输入一个任意形状的矩阵，输出矩阵U、R、V

7. solve_linear_system:输入方阵A，向量b。输出解向量x。矩阵A必须是方阵，保证有唯一解
8. square_det:输入方阵A的行列式值

# 5个运行案例
9. example_LU
10. example_QR_Gram_Schmidt
11. example_householder
12. example_givens
13. example_urv
"""


# 提供几个矩阵用于测试，包括任意形状的奇异矩阵和非奇异矩阵
def get_matrix(mode):
    if mode == 0:
        return [
            [2, 2, 2],
            [4, 7, 7],
            [6, 18, 22]
        ]
    if mode == 1:
        return [
            [0, -20, -14],
            [3, 27, -4],
            [4, 11, -2]
        ]
    if mode == 2:
        return [
            [1, 19, -34],
            [-2, -5, 20],
            [2, 8, 37]
        ]
        # b = [12, 24, 12]
    if mode==3:
        return [
            [2, 2, 2],
            [4, 7, 7],
            [8, 14, 14]
        ]
    if mode == 4:
        return [
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]
        ]

    if mode == 5:
        return [
            [1, 2, 1, 5, 1, 0, 0],
            [1, 2, 1, 5, 1, 0, 0],
            [1, 2, 1, 5, 0, 0, 0],
            [0, 0, 0, 0, 2, 3, 1],
            [0, 0, 0, 0, 2, 3, 1]
        ]


# 高斯消去，化为上三角阵，仅返回一个上三角阵
# 作用：用于求秩、求行列式、求线性方程组
def gauss_jordan(A):
    A = np.array(A)
    r, c = A.shape
    A = np.array(A).astype(np.float64)

    for j in range(c - 1):
        # 1. 首先找出绝对值最大的值，和主元位置交换行
        max_ind = j
        max_pivot = A[j][j]
        for ii in range(j, r):
            if np.abs(A[ii][j]) <= np.abs(max_pivot):
                continue
            else:
                max_pivot = A[ii][j]
                max_ind = ii
        if max_pivot == 0:
            print("0 pivot emerge LU failure...")
            return
        else:
            # print('pivot and index: ', max_pivot, max_ind)
            A[[max_ind, j], :] = A[[j, max_ind], :]
            # print('current A\n', A)

        # 2. 然后消去主元下方元素
        # print('cur, all', j + 1, r)
        for i in range(j + 1, r):
            coef = np.round(A[i, j] / max_pivot, 4)
            # print('elimination factor:', coef)
            for jj in range(j, c):
                # print(A[i, jj] - coef * A[j, jj], '=', A[i, jj], '-', coef, '*', A[j, jj])
                A[i, jj] = np.round(A[i, jj] - coef * A[j, jj], 3)
    return A


# LU Factorization
"""
input: np.array([[]]), a matrix
output: tuple(L, U)

"""


# 具体方法：Gauss-Jordan消去法
def luf(A):
    print("----LU Factorization----")
    A = np.array(A).astype(np.float64)
    print('matrix A: \n', A)
    r, c = A.shape
    print('shape: ', r, c)
    assert r == c, "A not a square, cannot be LU factorized..."  # 确保是一个方阵

    P = np.identity(c)
    L = np.identity(c)
    for j in range(c - 1):
        # 1. 首先找出绝对值最大的值，和主元位置交换行
        max_ind = j
        max_pivot = A[j][j]
        for ii in range(j, r):
            if np.abs(A[ii][j]) <= np.abs(max_pivot):
                continue
            else:
                max_pivot = A[ii][j]
                max_ind = ii
        if max_pivot == 0:
            # print("0 pivot emerge LU failure...")
            return
        else:
            # print('pivot and index: ', max_pivot, max_ind)
            P[[max_ind, j], :] = P[[j, max_ind], :]
            A[[max_ind, j], :] = A[[j, max_ind], :]
            # print('current A\n', A)

        # 2. 然后消去主元下方元素
        # print('cur, all', j + 1, r)
        for i in range(j + 1, r):
            coef = np.round(A[i, j] / max_pivot, 4)
            # print('elimination factor:', coef)
            for jj in range(j, c):
                # print(A[i, jj] - coef * A[j, jj], '=', A[i, jj], '-', coef, '*', A[j, jj])
                A[i, jj] = np.round(A[i, jj] - coef * A[j, jj], 3)
            L[i, j] = coef
            # print('current A\n', A)
    return L, A, P


# 采用modified gram-schmidt
# 正交化和QR分解都不要求方阵
def qr_gram_schmidt(A):
    print("----Gram Schmidt----")
    A = np.array(A)
    print('matrix A: \n', A)
    r, c = A.shape
    # 必需步骤，转换为浮点类型方能正常运算
    A = A.astype(np.float64)
    Q = A
    R = np.identity(c)
    # new_basis = new_basis.astype(np.float64)
    for j in range(c):
        # print(f"----k={j}----")
        # 获取一个新基
        uj = A[:, j]
        if np.sum(uj * uj) < 1e-6:
            continue
        # print(f"u{j}: ", uj)
        uj_norm2 = np.round(float(np.sqrt(np.sum(uj * uj))), 4)
        # print(f"u{j}_norm2: ", uj_norm2)
        R[j, j] = uj_norm2
        uj = uj / uj_norm2
        # print(f"u{j}: ", uj)
        Q[:, j] = uj
        # print(Q)
        # 更新其他新基
        for k in range(j + 1, c):
            uk = A[:, k]
            # print(f"u{k}", uk)
            rjk = np.round(float(np.sum(uj * uk)), 4)
            R[j, k] = rjk
            A[:, k] = uk - rjk * uj
            # print(f"u{k}", uk - np.round(float(np.sum(uj * uk)), 4) * uj)
            # print()
    # print("----Q----\n", Q)
    # print("----R----\n", R)
    return Q, R


def qr_householder(A):
    print("====householder=========")
    A = np.array(A)
    print("A:\n", A)
    r, c = A.shape
    A = A.astype(np.float64)
    # new_basis = A
    Q = np.identity(r)
    RA = A
    for j in range(0, c - 1):
        # print("----", j, "----")
        Aj = RA[j:, j]
        # print("Aj", Aj)
        uj = Aj
        if np.sum(uj * uj) < 1e-6:
            continue
        uj_norm = np.round(float(np.sqrt(np.sum(uj * uj))), 4)
        # print("uj_norm: ", uj_norm)
        ej = np.array([1] + [0 for _ in range(r - 1 - j)])
        # print("ej: ", ej)
        uj = np.round(uj - uj_norm * ej, 4)
        if np.sum(uj * uj) < 1e-6:
            continue
        # print("uj: ", uj)
        R = np.identity(r - j) - 2.0 / np.sum(uj * uj) * (uj.reshape((r - j, 1)) * uj)
        # print("R:\n", R)
        if j > 0:
            R = np.vstack([
                np.hstack([np.identity(j), np.zeros((j, r - j))]),
                np.hstack([np.zeros((r - j, j)), R])
            ])
            # print("3dim R:\n", R)
        Q = np.matmul(R, Q)
        RA = np.matmul(R, RA)
    #     print("RA:\n", np.round(RA, 4))
    # print("Q:\n", np.round(Q, 4))
    return Q.T, RA


def qr_givens(A):
    A = np.array(A)
    print("=======Givens======\nA:\n", A)
    r, c = A.shape
    Q = np.identity(r)
    A = A.astype(np.float64)
    R = A
    for j in range(c - 1):
        # print(f"========{j}=========")
        Aj = R[:, j]
        # print("Aj: ", Aj)
        Pj = np.identity(r)
        for i in range(j + 1, r):
            Pji = np.identity(r)
            deno = np.round(np.sqrt(Aj[j] * Aj[j] + Aj[i] * Aj[i]), 4)
            if deno < 1e-6:
                continue
            Pji[j, j] = 0
            Pji[i, i] = 0
            Pji[j, j] = Aj[j] / deno
            Pji[j, i] = Aj[i] / deno
            Pji[i, j] = -Aj[i] / deno
            Pji[i, i] = Aj[j] / deno
            # print("rotation Pji:\n", Pji)
            Aj = np.matmul(Pji, Aj)
            Pj = np.matmul(Pji, Pj)
        Q = np.matmul(Pj, Q)
        R = np.round(np.matmul(Pj, R), 4)
        # print("----Q----:\n", Q, "\n----R----:\n", R)
    return Q.T, R


def urv(A):
    print("=========URV======")
    Q1, R1 = qr_householder(A)
    Q2, R2 = qr_householder(R1.T)
    # Q1, R1 = qr_givens(A)
    # Q2, R2 = qr_givens(R1.T)
    return Q1, R2.T, Q2.T


# 求线性方程组Ax=b，采用高斯消去法，
# 虽然LU分解也可以，但是LU分解多计算了L矩阵和P矩阵
# 不如直接带等号右边项一起进行高斯消去，然后反向计算未知数x，更高效
def solve_linear_system(A, b):
    A = np.array(A)
    b = np.array(b)
    print("==========Solve Linear System=======")
    print("A:\n", A)
    print("b:\n", b)
    r, c = A.shape
    # 确保是一个方阵，这样才有唯一解
    assert r == c, "A not a square, cannot be LU factorized..."
    A = np.array(np.hstack([A, np.array(b).reshape(r, 1)])).astype(np.float64)
    A = gauss_jordan(A)
    res = 1
    for i in range(c):
        res *= A[i, i]
    if res < 1e-5:
        print("======不满秩，无解===Not full rank, no solution.....===")
        return
    # print("====计算解====")
    res = [0 for _ in range(c)]
    for i in range(c - 1, -1, -1):
        res[i] = A[i, c]
        s = str(res[i])
        for k in range(c - 1, i, -1):
            res[i] -= A[i, k] * res[k]
            s += " - (" + str(A[i, k]) + " x " + str(res[k])
        res[i] /= A[i, i]
        s += " ) / " + str(A[i, i])
        # print(s)
    return res


# 求行列式，使用高斯消去法求解
def square_det(A):
    A = np.array(A)
    print("=====det======\nA:\n", A)
    r, c = A.shape
    assert r == c, "A not a square, cannot be LU factorized..."  # 确保是一个方阵
    A = gauss_jordan(A)
    res = 1
    for i in range(c):
        res *= A[i, i]
    return res


def example_LU():
    print("=========LU分解案例===========")
    # A = np.random.randint(0, 100, size=(4, 4))
    L, U, P = luf(get_matrix(3))
    print('---L---\n', L)
    print('---U---\n', U)
    print('---P---\n', P)


def example_QR_Gram_Schmidt():
    print("=======gram-schmidt QR============")
    # A2 = np.array([
    #     [1 / 3, -2 / 3, 2 / 3],
    #     [-2 / 3, 1 / 3, 2 / 3],
    #     [2 / 3, 2 / 3, 1 / 3]
    # ])
    Q, R = qr_gram_schmidt(get_matrix(4))
    print("----Q----\n", np.round(Q, 3))
    print("----R----\n", np.round(R, 3))


def example_householder():
    Q, R = qr_householder(get_matrix(4))
    print("----Q----\n", np.round(Q, 3))
    print("----R----\n", np.round(R, 3))


def example_givens():
    Q, R = qr_givens(get_matrix(4))
    print("----Q----\n", np.round(Q, 3))
    print("----R----\n", np.round(R, 3))


def example_urv():
    A = get_matrix(2)
    U, R, V = urv(A)
    print("====原矩阵====\n", A)
    print("====分解所得====")
    print(U)
    print(R)
    print(V)
    print("====复原====")
    qr = np.matmul(U, np.matmul(R, V))
    # qr = np.matmul(np.matmul(U, R), V)
    print(qr)

    # print("====原矩阵====\n", A)
    # Q, R = qr_givens(A)
    # print("====分解所得====")
    # # 验证LU分解的正确性
    # print(Q)
    # print(R)
    # print("====复原====")
    # qr = np.matmul(Q, R)
    # print(qr)

    # L, U, P = luf(A)
    # print("====分解所得====")
    # print(L)
    # print(U)
    # print(P)
    # pa = np.matmul(P, A)
    # lu = np.matmul(L, U)
    # print("====复原====")
    # print(pa)
    # print(lu)


if __name__ == '__main__':
    print("input your example: ")
    s = input("0: lu, 1:grma_schmidt, 2: householder, 3: givens, 4: urv, \n5:solve linear system, 6: det\n\n")
    s = eval(s)
    if s == 0:
        example_LU()
    elif s==1:
        example_QR_Gram_Schmidt()
    elif s==2:
        example_householder()
    elif s==3:
        example_givens()
    elif s==4:
        example_urv()
    elif s==5:
        print(solve_linear_system(get_matrix(2), [12, 24, 12]))
    elif s==6:
        print(square_det(get_matrix(2)))
    print("====successful====")
