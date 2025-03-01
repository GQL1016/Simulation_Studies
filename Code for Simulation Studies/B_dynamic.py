# ╔═════════════════════════════════════════════════╗
# ║                 File Information                ║
# ╠═════════════════════════════════════════════════╣
# ║ Author: Gui qianlun                             ║
# ║ Date:   2024-09-01                              ║
# ╠═════════════════════════════════════════════════╣
# ║                 Environment Info                ║
# ╠═════════════════════════════════════════════════╣
# ║ Python version: 3.11.9                          ║
# ║ numpy: 2.1.0                                    ║
# ║ matplotlib: 3.9.2                               ║
# ║ cvxpy: 1.5.3                                    ║
# ║ joblib: 1.4.2                                   ║
# ╚═════════════════════════════════════════════════╝

import numpy as np
import math
import time
from A_generate_date import *


"""
                        =========================================================
                                 算法执行模块(Algorithm Execution Module)        
                        =========================================================
"""



""" 
                      Algorithm 2. Stage I: Initialization  
"""
# 论文理论: Algorithm 2 中低秩矩阵的真实秩 `r` 是已知的, 即 `TrueRankIner[0]`
# 代码实现: 用 ChooseComb 函数通过交叉验证选取`rhat`替代论文中低秩矩阵真实秩`r`, 用于迭代初始化
  # Sigmahat_X_group_cov_w:  时刻点 w 处, 控制组的协方差估计
  # Sigmahat_Y_group_cov_w:  时刻点 w 处，实验组的协方差估计
  # alphahat, betahat, shat, rhat: 调优参数

def OurInitial(Sigmahat_X_group_cov_w, Sigmahat_Y_group_cov_w , alphahat, betahat, shat, rhat):

    d = Sigmahat_X_group_cov_w.shape[0]



    Deltahat0 = np.linalg.inv(Sigmahat_X_group_cov_w) - np.linalg.inv(Sigmahat_Y_group_cov_w)
    Deltahat1 = dynamic_clime(Sigmahat_X_group_cov_w)- dynamic_clime(Sigmahat_Y_group_cov_w)


    JsDeltahat = HardTrun(Deltahat1, int(shat)) # 硬阈值截断操作, 强制转换 shat 为整数
    S0 = PropTrun(JsDeltahat, alphahat)   # 分散截断操作


    if rhat >= 1:
        R0 = Deltahat0 - S0
        barU0, r1hat = LtoU(R0, int(rhat), 0)  # 强制转换 rhat 为整数
        Const= np.linalg.norm(barU0, 2)
        Incobound0 = 2 * Const * (betahat * rhat / d) ** (1 / 2)
        U0 = ProjInco(barU0, Incobound0)
        L0 = UtoL(U0, r1hat)  # L0=U0 @ diag[E_{r1hat},E_{rhat-r1hat}}] @ U0^T
        return S0, U0, L0, r1hat
    else:
        return S0, np.zeros((d, 0)), np.zeros((d, d)), 0



"""
            Algorithm 1. Stage II: projected alternating gradient descent for solving (7) 
"""

# Sigmahat_X_group_cov_w:  时刻点 w 处, 控制组的协方差估计
# Sigmahat_Y_group_cov_w:  时刻点 w 处，实验组的协方差估计
# S0, U0, L0: 迭代初始值
# tSparse, tLowrank: 真实稀疏，真实低秩
# r1hat: 正惯性指数
# alphahat, betahat, shat: 调优参数
# eta1, eta2: 步长1, 步长2
# TrueError: 保存真实误差值
# ObjLoss: 保存目标函数值
# Result: 保存结果的类
# maxIter: 最大迭代次数
# eps: 精度误差
def OurConverg(Sigmahat_X_group_cov_w, Sigmahat_Y_group_cov_w , S0, U0, L0, tSparse, tLowrank, \
               r1hat, alphahat, betahat, shat, eta1, eta2, TrueError, ObjLoss, Result, maxIter, eps):

    d = Sigmahat_X_group_cov_w.shape[0]

    # 计算差异和rhat
    hatDiff = Sigmahat_Y_group_cov_w - Sigmahat_X_group_cov_w
    rhat = np.size(U0, 1)

    # 迭代
    k = 0
    Idend = 1 # Idend 变量用来表示算法是否收敛. 具体来说, Idend 的值为 1 表示算法尚未收敛,而值为 0 表示算法已经收敛

    Lambda0 = genLambda(rhat, r1hat)

    TimeStart = time.time()
    while k <= maxIter:  # maxIter: 最大迭代次数
        # 计算一些公共项
        Term1 = S0 + L0
        Term2 = np.matmul(np.matmul(Sigmahat_X_group_cov_w, Term1), Sigmahat_Y_group_cov_w)
        Term3 = np.matmul(np.matmul(Sigmahat_Y_group_cov_w, Term1), Sigmahat_X_group_cov_w)
        Term4 = np.matmul(U0, Lambda0)
        Term5 = np.matmul(Term2, Term4)
        Term6 = np.matmul(Term3, Term4)
        Term7 = np.matmul(hatDiff, Term4)
        Term8 = np.matmul(U0.T, U0)

        # 迭代 S
        S12 = S0 - eta1 * (0.5 * Term2 + 0.5 * Term3 - hatDiff)
        S13 = HardTrun(S12, int(shat))  # 强制转换 shat 为整数
        S1 = PropTrun(S13, alphahat)

        # 迭代 U
        if rhat > 0:
            Const = np.linalg.norm(U0, 2)
            Incobound = 2 * Const * (betahat * rhat / d) ** (1 / 2)
            U12 = U0 - eta2 * (Term5 + Term6 - 2 * Term7) - eta2 / 2 * (np.matmul(U0, Term8) \
                              - np.matmul(np.matmul(Term4, Term8), Lambda0))
            U1 = ProjInco(U12, Incobound)
        else:
            U1 = U0.copy()   # 使用深拷贝 U1=U0.copy()

        # 计算误差  successive epsilon (连续精度)
        succeps = MaxDist(S0, U0, S1, U1, r1hat, rhat)
        L1 = UtoL(U1, r1hat)
        TrueError = CalTrueErr(TrueError, S1, L1, tSparse, tLowrank)
        # TrueError是一个用于存储每次迭代误差值 [Err1, Err2, Err3 / Scale] 的列表,它最初在 OurTwoStageMethod 中
        # 创建,并在 CalTrueErr 和 OurConverg 函数中被不断更新,以记录算法的收敛过程和误差变化情况
        ObjLoss = LossSampCov(ObjLoss, Sigmahat_X_group_cov_w, Sigmahat_Y_group_cov_w, S1, U1, L1, r1hat)
        # 论文中 (7) 式的目标优化函数
        # ObjLoss 是一个用于存储每次迭代过程中计算得到的目标函数值的列表, 它最初在 OurTwoStageMethod 中创建,
        # 并在 LossSampCov 和 OurConverg 函数中被不断更新, 以记录算法的收敛过程和损失变化情况

        if ObjLoss[-1] > ObjLoss[-2]:  # 如果损失增加, 需要减小步长
            eta1 = eta1 / (k + 1)
            eta2 = eta2 / (k + 1)
        if succeps <= eps:
            Idend = 0  # Idend变量用来表示算法是否收敛. 具体来说, Idend 的值为 1 表示算法尚未收敛，而值为 0 表示算法已经收敛
            break

        S0 = S1.copy() # 深拷贝
        U0 = U1.copy() # 深拷贝
        L0 = L1.copy() # 深拷贝
        k = k + 1

    TimeEnd = time.time()

    # 保存结果
    Result.CostTime = TimeEnd - TimeStart  # 记录迭代运行的总时间
    Result.Idend = Idend  # 保存迭代是否收敛的状态, Idend 变量用来表示迭代是否收敛: 1 表示未收敛, 0 表示已收敛
    Result.TrueError = TrueError  #  用于存储每次迭代过程中计算得到的真实误差值 [Err1, Err2, Err3 / Scale] 的列表
    Result.ObjLoss = ObjLoss  # 用于存储每次迭代过程中计算得到的目标函数值的列表
    Result.Sparse = S1   # 保存最终的稀疏矩阵估计结果
    Result.Lowrank = L1  # 保存最终的低秩矩阵估计结果
    Result.RankInerLowrank =  LtoU(L1, 1e-3, 1)[1:]
    # 对最终的低秩矩阵 L1 进行特征值分解同时获取该矩阵的秩和正惯性指数,并将其按顺序保存到 Result. RankInerLowrank 中
    # (原代码中没有这项,自己添加的)
    Result.tSparse = tSparse   # 保存真实的稀疏矩阵,以便与估计结果进行对比,用于评估估计的准确性
    Result.tLowrank = tLowrank  # 保存真实的低秩矩阵,以便与估计结果进行对比,用于评估估计的准确性
    Result.IterCount = len(TrueError)  # 保存迭代的总次数, 即 TrueError 列表的长度

    return Result   # 返回保存了所有结果的 Result 对象



"""
         Implementing a two-stage algorithm, including the initialization stage and the convergence stage
"""

# 用于实现两阶段算法
  # w: 时刻点
  # X_group:  控制组的观测向量集
  # Y_group:  实验组的观测向量集
  # X_h2_opt: 控制组的最优带宽
  # Y_h2_opt: 实验组的最优带宽
  # tSparse, tLowrank: 真实稀疏，真实低秩
  # TrueRankIner: 一个包含两个整数的列表, 表示真实秩和正惯性指数
  # alphahat, betahat, shatprob, rhat: 调优参数
  # eta1: 步长1
  # maxIter: 最大迭代次数
  # eps: 阈值误差

def OurTwoStageMethod(w, Sigmahat_X_group_cov_w, Sigmahat_Y_group_cov_w, \
                      X_h2_opt, Y_h2_opt,\
                      tSparse, tLowrank, TrueRankIner, \
                      alphahat=0.5, betahat=1, shatprob=5, rhat=2, \
                      eta1=0.5, maxIter=5000, eps=1e-5):


    Result = SimuResult()
    Result.Point = w
    Result.TrueRankIner = TrueRankIner

    d = Sigmahat_X_group_cov_w.shape[0]

    shat = math.ceil(d * (1 + shatprob))  # shat 为稀疏部分非零元素总个数的上限, 强制转换为整数
    # 运行阶段-I算法
    S0, U0, L0, r1hat = OurInitial(Sigmahat_X_group_cov_w, Sigmahat_Y_group_cov_w, alphahat, betahat, shat, rhat)

    Result.Param = [alphahat, betahat, shatprob, rhat, r1hat]

    # 初始化误差矩阵
    TrueError, ObjLoss = [], []
    TrueError = CalTrueErr(TrueError, S0, L0, tSparse, tLowrank)
    ObjLoss = LossSampCov(ObjLoss, Sigmahat_X_group_cov_w,  Sigmahat_Y_group_cov_w, S0, U0, L0, r1hat)

    # 运行阶段-II算法
    # 指定 eta2 使得 0.2 <= eta2 <= 0.7
    # 对于我们的方法, 步长设置为 $\eta_1 = 0.5$ , $\eta_2 = \eta_1 / \sigma_{\max}^2(U^0)$, 其中 $U^0$ 是初始化步骤的输出
    if  rhat > 0 :
        if 3 * eta1 <= np.linalg.norm(U0, 2) ** 2 and np.linalg.norm(U0, 2) ** 2 <= 5 * eta1:
            eta2 = eta1 / np.linalg.norm(U0, 2) ** 2
        else:
            eta2 = eta1 / 2
    else:
        eta2 = 1

    Result = OurConverg(Sigmahat_X_group_cov_w, Sigmahat_Y_group_cov_w, S0, U0, L0, tSparse, tLowrank, \
                        r1hat, alphahat, betahat, shat, eta1, eta2, TrueError, ObjLoss, Result, maxIter, eps)


    return Result



# 类定义, 用于模拟结果
class SimuResult:    # 定义的类: SimuResult 通过 OurTwoStageMethod 函数将其结构赋予 Result
    def __init__(self):
        self.Point = 0  # 时刻点 w
        self.Param = [0, 0, 0, 0, 0]  # 参数设置：alphahat,betahat, shatprob, rhat, r1hat. OurTwoStageMethod+
        self.CostTime = 0  # 花费时间. OurConverg+
        self.Idend = 1  # Idend变量用来表示算法是否收敛,具体来说, Idend 的值为 1 表示算法尚未收敛,而值为 0 表示算法已经收敛. OurConverg+
        self.TrueError = []  # 真实误差值. OurConverg+
        self.ObjLoss = []  # 目标函数值. OurConverg+
        self.Sparse = np.zeros(0)    # 输出: 最终估计的稀疏成分.  OurConverg+
        self.Lowrank = np.zeros(0)   # 输出: 最终估计的低秩成分.  OurConverg+
        self.RankInerLowrank = [0,0] # 输出: 最终估计的低秩成分的秩和正惯性指数. OurConverg+ (自己添加的,原代码没有)
        self.tSparse = np.zeros(0)   # 真实的稀疏成分.  OurConverg+
        self.tLowrank = np.zeros(0)  # 真实的低秩成分. OurConverg+
        self.IterCount = 0  # 迭代次数. OurConverg+
        self.TrueRankIner = [0, 0]  # 真实秩和正惯性指数.  OurTwoStageMethod+





"""
                                   Required Auxiliary Functions B
"""

# 根据训练集和测试集,计算损失值
# Covhat_TestX, Covhat_TestY:  控制组测试集的样本协方差估计, 实验组测试集的样本协方差估计
# Result: 结果类
def Loss(Covhat_TestX, Covhat_TestY, Result):
    DiffCov = Covhat_TestY - Covhat_TestX
    DeltaHat = Result.Sparse + Result.Lowrank # 最终估计的稀疏成分 + 最终估计的低秩成分
    A = np.matmul(DeltaHat, np.matmul(Covhat_TestX, np.matmul(DeltaHat, Covhat_TestY)))
    B = np.matmul(DeltaHat, DiffCov)
    return 0.5 * np.trace(A) - np.trace(B)   # 论文中的(5)式: empirical loss


# 该函数根据给定的低秩矩阵 L 分解出 U: L = U @ Lambda @ U^T
  # L: 低秩成分
  # rhatThres: 可以是rhat或阈值
  # Id:  Id = 0: 给定rhat
  #      Id = 1: 给定阈值
def LtoU(L, rhatThres, Id):
    EigVal, EigVec = np.linalg.eig(L) # 计算特征值和特征向量
    if Id == 0:  # Id = 0: 给定rhat

        rhatThres = min(int(rhatThres), np.size(L, 0))  # 强制转换rhatThres为整数
        sorted_indices = np.argsort(np.abs(EigVal))[::-1] # 获取按绝对值降序排列的特征值索引
        top_indices = sorted_indices[:rhatThres] # 获取前 rhatThres 个最大的特征值索引

        EigVal = EigVal[top_indices]    # 重新筛选特征值和特征向量
        EigVec = EigVec[:, top_indices] # 从矩阵 EigVec 中选择出 top_indices 对应的列

        IdEig = EigVal.argsort()[::-1]
        EigVal = EigVal[IdEig]
        EigVec = EigVec[:, IdEig]
        EigVec = EigVec.real


        U = np.matmul(EigVec, np.diag((np.abs(EigVal)) ** (1 / 2)))
        r1 = sum(EigVal > 0)

        return U, r1

    elif Id == 1: # Id = 1: 给定阈值   # 在 MyMain 中对真实低秩矩阵进行特征值分解，以求解矩阵的秩和正惯性指数.

        IdEig = (np.abs(EigVal) >= rhatThres)
        EigVal = EigVal[IdEig]
        EigVec = EigVec[:, IdEig]
        IdEig = EigVal.argsort()[::-1]
        EigVal = EigVal[IdEig]
        EigVec = EigVec[:, IdEig]
        EigVec = EigVec.real
        U = np.matmul(EigVec, np.diag((np.abs(EigVal)) ** (1 / 2)))
        r = len(EigVal)
        r1 = sum(EigVal > 0)

        return U, r, r1



# 对实对称矩阵 A 按给定的 s 进行硬阈值截断操作
def HardTrun(A, s):
    n = A.shape[0]
    assert A.shape[1] == n, "矩阵A必须是方阵"

    # 第一步: 选择上三角部分（不包括对角线）最大的前 \tilde{s} 个元素
    UOffM = np.triu(A, 1)
    abs_u_elements = np.abs(UOffM[UOffM != 0])

    # 修改的部分: 计算 tilde_s
    tilde_s = math.ceil(s / 2)
    if tilde_s > len(abs_u_elements):
        tilde_s = len(abs_u_elements)
    largest_u_indices = np.argsort(-abs_u_elements)[:tilde_s]
    largest_u_values = abs_u_elements[largest_u_indices]

    # 第二步: 选择对角线部分最大的前 n 个元素
    DiagM = np.diag(A)
    abs_d_elements = np.abs(DiagM)
    largest_d_indices = np.argsort(-abs_d_elements)[:n]
    largest_d_values = abs_d_elements[largest_d_indices]

    # 第三步: 混合两个序列并降序排列
    mixed_values = np.concatenate((largest_u_values, largest_d_values))
    sorted_mixed_indices = np.argsort(-mixed_values)
    sorted_mixed_values = mixed_values[sorted_mixed_indices]

    # 第四步: 选择前 s 个元素
    selected_indices = []
    count = 0
    for value in sorted_mixed_values:
        if value in largest_u_values:
            count += 2
        elif value in largest_d_values:
            count += 1
        if count > s+1:
            break
        selected_indices.append(value)

    # 创建结果矩阵并截断非选中元素
    result_matrix = np.zeros_like(A)
    for value in selected_indices:
        if value in largest_u_values:
            index = np.where(abs_u_elements == value)[0][0]
            i, j = np.triu_indices(n, 1)[0][index], np.triu_indices(n, 1)[1][index]
            result_matrix[i, j] = A[i, j]
            result_matrix[j, i] = A[j, i]
            largest_u_values = np.delete(largest_u_values, np.where(largest_u_values == value))
        elif value in largest_d_values:
            index = np.where(abs_d_elements == value)[0][0]
            i = index
            result_matrix[i, i] = A[i, i]
            largest_d_values = np.delete(largest_d_values, np.where(largest_d_values == value))

    return result_matrix



# 对实对称矩阵 A 按给定的 alpha 进行分散阈值截断操作
def PropTrun(A, alpha):

    d = A.shape[0]
    tilde_s = math.ceil(alpha * d)

    # 第一步: 选择每列中绝对值最大的 alpha * d 个元素
    col_indices_set = set()
    for j in range(d):
        col_indices = np.argsort(-np.abs(A[:, j]))[:tilde_s]
        for i in col_indices:
            col_indices_set.add((i, j))

    # 第二步: 选择每行中绝对值最大的 alpha * d 个元素
    row_indices_set = set()
    for i in range(d):
        row_indices = np.argsort(-np.abs(A[i, :]))[:tilde_s]
        for j in row_indices:
            row_indices_set.add((i, j))

    # 第三步: 取交集
    final_indices_set = col_indices_set.intersection(row_indices_set)

    # 创建结果矩阵并保留交集位置的元素, 其余位置设为 0
    result_matrix = np.zeros_like(A)
    for (i, j) in final_indices_set:
        result_matrix[i, j] = A[i, j]
        result_matrix[j, i] = A[j, i]  # 确保对称性

    return result_matrix



# 低秩矩阵的投影函数, 以满足非一致性条件
  # M: d x r 矩阵
  # B: 非一致性条件的界限
  # 注: ProjInco 函数实际上是在调整矩阵 M 的行，对矩阵 M 的行向量进行缩放,以确保每一行的范数满足指定的非一致性条件
def ProjInco(M, B):
    RowNorm = np.linalg.norm(M, axis=1)
    Idnorm = RowNorm > B
    RowNorm[Idnorm] = B / RowNorm[Idnorm]
    RowNorm[(1 - Idnorm).astype(bool)] = 1
    return np.matmul(np.diag(RowNorm), M)



# 根据 U 和 r1 计算低秩矩阵的函数
  # U:  低秩因子
  # r1: 正惯性指数
def UtoL(U, r1):
    r = np.size(U, 1)
    Lambda = genLambda(r, r1)
    L = np.matmul(U, np.matmul(Lambda, U.T))
    return L


# 生成 Lambda 矩阵的函数, 给定 r1 和 r
  # r:  秩
  # r1: 正惯性指数 (r1 <= r)
def genLambda(r, r1):
    Lambda = np.concatenate((np.concatenate((np.eye(r1), np.zeros((r1, r - r1))), axis=1), \
                        np.concatenate((np.zeros((r - r1, r1)), -1 * np.eye(r - r1)), axis=1)), axis=0)
    return Lambda



# 定义总误差距离的函数
# S0, U0: 上一对
# S1, U1: 下一对
# r1hat: 正惯性指数
# rhat:  秩
def MaxDist(S0, U0, S1, U1, r1hat, rhat):
    if rhat == 0:
        Scale = 1
    else:
        Scale = np.linalg.norm(U1, 2) ** 2

    # D1: 计算稀疏部分连续迭代之间的误差 $\|S_0-S_1\|_{F}^{2}/(\sigma^{U1}_{\text{max}})^{2}$.
    D1 = np.linalg.norm(S0 - S1, 'fro') ** 2 / Scale

    # D2: 计算低秩部分连续迭代之间的误差 $\Pi^{2}(U_0,U_1)$
    # $\Pi_{\hat{r}_1}(U_0, U_1) = \inf_{Q \in Q^{\hat{r} \times \hat{r}}_{\hat{r}_1}} \| U_1 - U_2 Q \|_F,$
    # $\begin{align*}
    #  Q^{\hat{r} \times \hat{r}}_{\hat{r}_1} &= \{ Q \in \mathbb{Q}^{\hat{r} \times \hat{r}} : Q \Lambda Q^T = \Lambda \quad\text{其中}\quad \Lambda = \text{diag}(I_{\hat{r}_1}, -I_{\hat{r}-\hat{r}_1}) \} \\
    #  &= \{ Q \in \mathbb{Q}^{\hat{r} \times \hat{r}} : Q = \text{diag}(Q_1, Q_2) \quad\text{其中}\quad  Q_1 \in \mathbb{Q}^{\hat{r}_1 \times \hat{r}_1}, Q_2 \in \mathbb{Q}^{(\hat{r}-\hat{r}_1)} \times (\hat{r}-\hat{r}_1) \}.
    #  \end{align*}$
    U01 = U0[:, 0:r1hat]
    U02 = U0[:, r1hat:]
    U11 = U1[:, 0:r1hat]
    U12 = U1[:, r1hat:]
    UU1 = np.matmul(U01.T, U11)
    UU2 = np.matmul(U02.T, U12)
    if r1hat > 0:
        D21 = np.linalg.norm(U01) ** 2 + np.linalg.norm(U11) ** 2 - 2 * np.linalg.norm(UU1, 'nuc')
    else:
        D21 = 0

    if rhat - r1hat > 0:
        D22 = np.linalg.norm(U02) ** 2 + np.linalg.norm(U12) ** 2 - 2 * np.linalg.norm(UU2, 'nuc')
    else:
        D22 = 0
    D2 = D21 + D22

    return max(D1, D2)



# 计算真实误差的函数
  # TrueError: 保存误差的列表
  # S0, L0:    一对稀疏和低秩矩阵
  # tSparse, tLowrank: 真实的稀疏和低秩矩阵
def CalTrueErr(TrueError, S0, L0, tSparse, tLowrank):

    Err1 = np.linalg.norm(S0 - tSparse, 'fro')
    Err2 = np.linalg.norm(L0 - tLowrank, 'fro')
    Err3 = np.linalg.norm(tSparse + tLowrank - (S0 + L0), 'fro')
    if np.linalg.norm(tLowrank,'fro') < 1e-6:
        Scale = 1
    else:
        Scale = np.linalg.norm(tLowrank, 2) ** (1 / 2)
    TrueError.append([Err1, Err2, Err3 / Scale])
    # 我们通过 $\|\hat{S} - S^\star\|_F$ 和 $\|\hat{\Delta} - \Delta^\star\|_F / \sqrt{\sigma_{\max}(R^\star)}$
    # 来衡量性能, 后者用作总误差距离 $TD(\hat{S},\hat{U})$ 的替代
    return TrueError



# 计算样本协方差的损失函数
  # ObjLoss: 保存损失的列表
  # hatSigmaX, hatSigmaY: 两组的样本协方差估计
  # S0, U0, L0, r1hat: 稀疏和低秩矩阵
def LossSampCov(ObjLoss, hatSigmaX, hatSigmaY, S0, U0, L0, r1hat):
    DeltaHat = S0 + L0
    hatDiff = hatSigmaY - hatSigmaX
    A = np.linalg.norm(np.matmul(U0[:, 0:r1hat].T, U0[:, r1hat:]), 'fro') ** 2 / 2
    B = np.matmul(DeltaHat, np.matmul(hatSigmaX, np.matmul(DeltaHat, hatSigmaY)))
    C = np.matmul(DeltaHat, hatDiff)
    D = 0.5 * np.trace(B) - np.trace(C) + A
    # 论文中 (7) 式的目标优化函数,其中 0.5*np.trace(B)-np.trace(C) 为论文中 (5)&(6) 公式
    ObjLoss.append(D)
    return ObjLoss
