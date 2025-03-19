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

import os
import time
import numpy as np
import matplotlib

from A_generate_date import *
from B_dynamic import *


matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置全局随机种子
global_seed = 2025
# 使用全局随机种子初始化NumPy的全局随机数生成器, 使其在每次程序执行时产生相同的随机数序列
np.random.seed(global_seed)
# split_counter = 0  #  不使用并行 joblib 的方案时才使用


"""
                     ==============================================================================

                        Module B (Dynamic): 基于模型 (控制组和实验组) 生成观测数据,并计算最优带宽等有关结果

                     ==============================================================================
"""

# 开始统计总用时
total_start_time = time.time()

Control_Idm = 0
Trial_Idm = 1
d = 60
r = 2
n = 1800


print(f"状态: Dynamic / 控制组: {Control_Idm} - 实验组: {Trial_Idm} / 观测向量维度: {d} / 隐变量个数: {r} / 观测向量个数: {n}")
print()  # 这将打印一个空行


# 定义采样时间序列的函数
def sample_time_series(Idm, n, global_seed):
    # 创建用于模型 Idm 随机数生成的生成器，种子是全局种子加上 Idm
    rng = np.random.default_rng(global_seed + Idm)
    # 使用模型 Idm 的随机数生成器生成 [0, 1] 区间内服从均匀分布的样本时间序列
    u_values = rng.uniform(0, 1, n)
    return u_values


X_u_values = sample_time_series(Control_Idm, n, global_seed)
Y_u_values = sample_time_series(Trial_Idm, n, global_seed)


def h_min_max(u_values):
    data_mad = np.median(np.abs(u_values - np.median(u_values)))
    h_min = 1 / 3 * data_mad   # 最近一次修改4: 1/3 改 1/2
    h_max = 3 * data_mad       # 最近一次修改4: 3 改 2
    return h_min, h_max


X_h_min, X_h_max = h_min_max(X_u_values)
print(f"X_h_min: {X_h_min:.6f} / X_h_max: {X_h_max:.6f}")
Y_h_min, Y_h_max = h_min_max(Y_u_values)
print(f"Y_h_min: {Y_h_min:.6f} / Y_h_max: {Y_h_max:.6f}")


start_time = time.time()
# 根据控制组的样本时间序列 X_u_values 一 一 对应地生成控制组的观测向量序列
X_combined_vectors = generate_combined_vectors(Control_Idm, d, r, X_u_values, global_seed)
# 提取控制组的观测向量序列, 跳过每个向量中的第一个元素 (即 u 值)
X_obs_vectors = X_combined_vectors[:, 1:]  # 取所有行的第二列到最后一列
end_time = time.time()
print(f"获取 控制组(Idm={Control_Idm}) {n} 个观测向量的执行时间: {end_time - start_time:.3f} 秒")


start_time = time.time()
# 根据实验组的样本时间序列 Y_u_values 一 一 对应地生成实验组的观测向量序列
Y_combined_vectors = generate_combined_vectors(Trial_Idm, d, r, Y_u_values, global_seed)
# 提取实验组的观测向量序列, 跳过每个向量中的第一个元素 (即 u 值)
Y_obs_vectors = Y_combined_vectors[:, 1:]  # 取所有行的第二列到最后一列
end_time = time.time()
print(f"获取 实验组(Idm={Trial_Idm}) {n} 个观测向量的执行时间: {end_time - start_time:.3f} 秒")
print()  # 这将打印一个空行


start_time = time.time()
X_h1_opt, X_min_mse, X_h1_vals, X_mse_vals = \
    select_optimal_bandwidth1(X_u_values, X_obs_vectors, X_h_min, X_h_max)
end_time = time.time()
print(f"选择 控制组(Idm={Control_Idm}) h1_opt 的执行时间: {end_time - start_time:.3f} 秒")
print(f"控制组(Idm={Control_Idm}) / h1_opt = {X_h1_opt} / h1_opt 处的最优 MSE 值: {X_min_mse}")


start_time = time.time()
Y_h1_opt, Y_min_mse, Y_h1_vals, Y_mse_vals = \
    select_optimal_bandwidth1(Y_u_values, Y_obs_vectors, Y_h_min, Y_h_max)
end_time = time.time()
print(f"选择 实验组(Idm={Trial_Idm}) h1_opt 的执行时间: {end_time - start_time:.3f} 秒")
print(f"实验组(Idm={Trial_Idm}) / h1_opt = {Y_h1_opt} / h1_opt 处的最优 MSE 值: {Y_min_mse}")
print()  # 这将打印一个空行


start_time = time.time()
X_h2_opt, X_min_cv, X_h2_vals, X_cv_vals = \
    select_optimal_bandwidth2(X_u_values, X_obs_vectors, X_h_min, X_h_max, X_h1_opt)
end_time = time.time()
print(f"选择 控制组(Idm={Control_Idm}) h2_opt 的执行时间: {end_time - start_time:.3f} 秒")
print(f"控制组(Idm={Control_Idm}) / h2_opt = {X_h2_opt} / h2_opt 处的最优 CV 值 : {X_min_cv}")


start_time = time.time()
Y_h2_opt, Y_min_cv, Y_h2_vals, Y_cv_vals = \
    select_optimal_bandwidth2(Y_u_values, Y_obs_vectors, Y_h_min, Y_h_max, Y_h1_opt)
end_time = time.time()
print(f"选择 实验组(Idm={Trial_Idm}) h2_opt 的执行时间: {end_time - start_time:.3f} 秒")
print(f"实验组(Idm={Trial_Idm}) / h2_opt = {Y_h2_opt} / h2_opt 处的最优 CV 值: {Y_min_cv}")
print()  # 这将打印一个空行


# 结束总用时统计
total_end_time = time.time()


total_ModuleB_time = total_end_time - total_start_time
print(f"Module B (Dynamic) 总执行时间: {total_ModuleB_time:.4f} 秒")
print()  # 这将打印一个空行



"""

                                   Required Auxiliary Functions A (Dynamic)

"""


# 计算 真实稀疏部分 tSparse 和 真实低秩序部分 tLowrank
def compute_tSparse_tLowrank(Control_Idm, Trial_Idm, d, r, u, global_seed):
    # 将 u 从 [-1, 1] 映射到 [0, 10^6] 区间
    mapped_u = int((u + 1) * 5e5)

    # 使用 global_seed 和 mapped_u 为控制组创建一个独立的生成器
    rng_control = np.random.default_rng(global_seed + mapped_u)
    # 生成控制组的矩阵组件
    Xpart1, Xpart2, Xpart3, Xpart4 = dynamic_joint_precmtx(Control_Idm, d, r, u,  rng_control)

    # 使用 global_seed 和 mapped_u 为实验组创建另一个独立的生成器
    rng_trial = np.random.default_rng(global_seed + mapped_u)
    # 生成实验组的矩阵组件
    Ypart1, Ypart2, Ypart3, Ypart4 = dynamic_joint_precmtx(Trial_Idm, d, r, u, rng_trial)

    # 计算 tSparse 和 tLowrank
    tSparse = Xpart1 - Ypart1
    tLowrank = -(Xpart2 @ np.linalg.inv(Xpart4) @ Xpart3 - Ypart2 @ np.linalg.inv(Ypart4) @ Ypart3)

    return tSparse, tLowrank





"""
                 ================================================================================

                                 Module C (Dynamic): 基于时刻点 w , 选取与之匹配的最优的参数组合

                 ================================================================================
"""



from joblib import Parallel, delayed
import numpy as np

# 并行处理的任务
def process_comb(w, TrueRankIner, X_h2_opt, Y_h2_opt,
                 alphahat, betahat, shatprob, rhat, split_pairs):

    loss_values = []  # 存储当前参数组合下的所有 Loss 值


    # 打印当前参数组合，便于跟踪
    print_info = f"[Dynamic w={w}] 参数组合: [alphahat={alphahat}, betahat={betahat}, shatprob={shatprob}, rhat={rhat}]"

    # 遍历数据对，分别计算 Loss
    for X_traingroup, X_testgroup, Y_traingroup, Y_testgroup in split_pairs:
        # 计算训练组和测试组的协方差矩阵
        Sigmahat_X_traingroup_cov_w = Sigmahat_cov(w, X_h2_opt, X_traingroup[:, 0], X_traingroup[:, 1:])
        Sigmahat_X_testgroup_cov_w = Sigmahat_cov(w, X_h2_opt, X_testgroup[:, 0], X_testgroup[:, 1:])
        Sigmahat_Y_traingroup_cov_w = Sigmahat_cov(w, Y_h2_opt, Y_traingroup[:, 0], Y_traingroup[:, 1:])
        Sigmahat_Y_testgroup_cov_w = Sigmahat_cov(w, Y_h2_opt, Y_testgroup[:, 0], Y_testgroup[:, 1:])

        '''
            注意:
            我们关心的是哪个参数组合在测试集上的经验损失 (empirical loss) 最小, 即 Loss 函数值最小.
            而利用Loss函数计算经验损失 (empirical loss) 时, 只需要知道最终估计的稀疏成分 Result.Sparse
            和最终估计的稀疏成分 Result.Lowrank 即可(不需要 OurConverg 函数迭代过程的 TrueError),
            所以并不需要给 OurTwoStageMethod 函数传入真实的 tSparse 和 tLowrank, 这里我们都传入 np.zeros((d, d)).
            使用零矩阵作为占位符, 可以减少不必要的计算负担, 提高参数选择的效率, 同时保证了参数选择过程的有效性          
        '''

        # 使用 A 训练, B 测试
        Result1 = OurTwoStageMethod(w, Sigmahat_X_traingroup_cov_w, Sigmahat_Y_traingroup_cov_w,
                                    X_h2_opt, Y_h2_opt,
                                    np.zeros((X_traingroup.shape[1] - 1, X_traingroup.shape[1] - 1)),
                                    np.zeros((Y_traingroup.shape[1] - 1, Y_traingroup.shape[1] - 1)),
                                    TrueRankIner, alphahat, betahat, shatprob, rhat)

        # Idend 变量用来表示算法是否收敛. 具体来说, Idend 的值为 1 表示算法尚未收敛, 而值为 0 表示算法已经收敛
        if Result1.Idend == 0:
            testloss1 = Loss(Sigmahat_X_testgroup_cov_w, Sigmahat_Y_testgroup_cov_w, Result1)
            loss_values.append(testloss1)

        # # 使用 B 训练, A 测试
        Result2 = OurTwoStageMethod(w, Sigmahat_X_testgroup_cov_w, Sigmahat_Y_testgroup_cov_w,
                                    X_h2_opt, Y_h2_opt,
                                    np.zeros((X_traingroup.shape[1] - 1, X_traingroup.shape[1] - 1)),
                                    np.zeros((Y_traingroup.shape[1] - 1, Y_traingroup.shape[1] - 1)),
                                    TrueRankIner, alphahat, betahat, shatprob, rhat)

        # Idend 变量用来表示算法是否收敛. 具体来说, Idend 的值为 1 表示算法尚未收敛, 而值为 0 表示算法已经收敛
        if Result2.Idend == 0:
            testloss2 = Loss(Sigmahat_X_traingroup_cov_w, Sigmahat_Y_traingroup_cov_w, Result2)
            loss_values.append(testloss2)

        '''
           注意:
           在 ChooseComb 函数中,只有收敛的迭代结果 (Result.Idend == 0) 才会被用于计算测试损失,并参与交叉验证和参数选择.
           未收敛的迭代结果 (Result.Idend = 1) 会被记录但不用于交叉验证参数选择.最终通过比较所有收敛结果的测试损失,
           选择最优的参数组合
        '''

    # 计算平均损失并保存参数组合及其损失值和相关信息
    if len(loss_values) > 0:
        average_loss = sum(loss_values) / len(loss_values)
        return [alphahat, betahat, shatprob, rhat, average_loss], print_info
    else:
        return [alphahat, betahat, shatprob, rhat, np.inf], print_info




# 主函数: 基于 w 选取与之匹配的最优参数组合 [alphahat, betahat, shatprob, rhat]
def ChooseComb(w, TrueRankIner, X_combined_vectors, Y_combined_vectors, X_h2_opt, Y_h2_opt,
               random_split_times = 2, \
               Alphahat=[0.01, 0.03, 0.05, 0.5], Betahat=[1, 3], Shatprob=[1, 3, 5], Rhat=[0, 1, 2, 3, 4]):


    split_counter = int((w + 1) * 5e5)  # 初始化 split_counter, 用于设置数据分割的随机种子

    total_split_count = 0  # 用于汇总所有的 split_combined_vectors 调用次数

    # 预先生成 int(random_split_times) 个随机均分观测数据对
    split_pairs = []

    for _ in range(int(random_split_times)):
        # 使用 split_counter 分割 X_combined_vectors
        X_traingroup, X_testgroup, split_counter = split_combined_vectors(np.copy(X_combined_vectors), split_counter)
        # 使用更新后的 split_counter 分割 Y_combined_vectors
        Y_traingroup, Y_testgroup, split_counter = split_combined_vectors(np.copy(Y_combined_vectors), split_counter)

        split_pairs.append((np.copy(X_traingroup), np.copy(X_testgroup),
                           np.copy(Y_traingroup), np.copy(Y_testgroup)))

    tasks = []

    '''
    注意:
    (1) 任务是按照如下循环的顺序来创建的.具体来说, alphahat, betahat, shatprob, rhat 
        是按照如下嵌套循环的顺序依次生成和匹配的
    (2) 虽然任务是按照顺序生成的,但在并行执行时,任务的执行顺序可能会与生成顺序不同.
        任务的执行顺序取决于并行库的调度机制以及系统的资源分配
    '''

    # 任务生成的顺序: 对每一个参数组合进行处理
    for alphahat in Alphahat:
        for betahat in Betahat:
            for shatprob in Shatprob:
                for rhat in Rhat:
                    tasks.append(
                        delayed(process_comb)(w, TrueRankIner, X_h2_opt, Y_h2_opt,
                                              alphahat, betahat, shatprob, rhat, split_pairs))
                    total_split_count += int(random_split_times)
                    # 每次循环调用 int(random_split_times) 次 split_combined_vectors

    # 并行执行所有任务
    results = Parallel(n_jobs=-1)(tasks)

    # 遍历 results 列表, results 列表中的每个元素都是一个二元组 (结果列表 result, 打印信息 print_info )
    # 在并行执行过程中, 任务的完成顺序可能与生成顺序不同, 但输出的顺序仍然会与任务生成的顺序一致
    for result, print_info in results:
        print(f"{print_info}: 结果 = {result}")
        # 打印出参数组合信息 (print_info) 和与之对应的计算结果 (result)

    # 从 results 列表中提取每个元组的第一个元素 (即 result 部分), 并将所有 result 列表组合成一个 NumPy 数组 Param
    Param = np.array([result[0] for result in results])

    # 选择损失最小的参数组合
    if np.all(Param[:, 4] == np.inf):  # 如果所有的结果都是 np.inf
        SelectParam = np.array([4, 0.15, 0.3, 4])
    else:
        SelectParam = Param[np.argmin(Param[:, 4]), 0:4]

    return SelectParam, total_split_count


"""
                 ====================================================================================

                               Module D (Dynamic): 针对不同时刻点 w , 评估真实与估计的各项差异并汇总结果

                 ====================================================================================
"""


import numpy as np

def OurApproach(Control_Idm, Trial_Idm, d, r, \
                X_combined_vectors, Y_combined_vectors, X_h2_opt, Y_h2_opt, global_seed = 2025):

    FinalResult = []  # 该列表只用于存储迭代收敛(Result.Idend == 0)的实验结果.



    # 选取 0 到 1 之间需要评估的点集
    w_values = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    for w in w_values:

        tSparse_w, tLowrank_w = compute_tSparse_tLowrank(Control_Idm, Trial_Idm, d, r, w, global_seed)
        tU_w, tr_w, tr1_w = LtoU(tLowrank_w, 1e-3, 1)
        TrueRankIner_w = [tr_w, tr1_w]

        Sigmahat_X_cov_w = Sigmahat_cov(w, X_h2_opt, np.copy(X_combined_vectors)[:, 0],\
                                        np.copy(X_combined_vectors)[:, 1:])
        Sigmahat_Y_cov_w = Sigmahat_cov(w, Y_h2_opt, np.copy(Y_combined_vectors)[:, 0],\
                                        np.copy(Y_combined_vectors)[:, 1:])

        selected_params_w, total_split_count_w = ChooseComb(w, TrueRankIner_w, \
                        np.copy(X_combined_vectors), np.copy(Y_combined_vectors), X_h2_opt, Y_h2_opt)

        # selected_params_w 中参数的顺序是 [alphahat, betahat, shatprob, rhat]
        print_info = f"[Dynamic w={w}] 最优参数组合: " \
                     f"[alphahat={selected_params_w[0]}, " \
                     f"betahat={selected_params_w[1]}, " \
                     f"shatprob={selected_params_w[2]}, " \
                     f"rhat={selected_params_w[3]}]"
        # 打印当前参数组合
        print(print_info)
        print()

        # 运行两阶段算法
        Result_w = OurTwoStageMethod(w, Sigmahat_X_cov_w, Sigmahat_Y_cov_w, \
                   X_h2_opt, Y_h2_opt,\
                   tSparse_w, tLowrank_w,\
                   TrueRankIner_w, \
                   selected_params_w[0], selected_params_w[1], selected_params_w[2], selected_params_w[3])

        if  Result_w.Idend == 0:
            # Idend 变量用来表示迭代是否收敛. 具体来说,Idend 的值为 1 表示算法尚未收敛, 而值为 0 表示算法已经收敛
            FinalResult.append(Result_w)
            # 如果算法收敛 (Idend == 0),保存结果. 即该列表只用于存储迭代收敛 (Result.Idend == 0) 的实验结果

    # 汇总模拟结果
    SumRes = SummarySimuResult()  # 创建汇总模拟结果的对象

    # 遍历每次实验结果,计算统计信息
    for result in FinalResult:

        SumRes.Point.append(result.Point)
        SumRes.ChooseParamsComb.append(result.Param)
        SumRes.IterCount.append(result.IterCount)  # 保存每次实验的迭代次数
        SumRes.CostTime.append(result.CostTime)  # 保存每次实验的耗时

        R1 = (result.TrueRankIner[1] == int(result.RankInerLowrank[1]))
        # 判断恢复的正惯性指数与真实的正惯性指数是否相等
        SumRes.TrueR1_Recovery.append(R1)  # 保存每次实验是否正确恢复正惯性指数的布尔值

        R = (result.TrueRankIner[0] == int(result.RankInerLowrank[0]))
        # 判断恢复的秩是否与真实的秩相等
        SumRes.TrueR_Recovery.append(R)  # 保存每次实验是否正确选择秩的布尔值

        TPR = calculate_TPR(result.Sparse, result.tSparse, threshold=1e-3)
        SumRes.TPR.append(TPR)

        FPR = calculate_FPR(result.Sparse, result.tSparse, threshold=1e-3)
        SumRes.FPR.append(FPR)

        SumRes.SparseFrobError.append(np.linalg.norm(result.Sparse - result.tSparse,'fro'))  # 保存每次实验的稀疏误差
        SumRes.SparseSpecError.append(np.linalg.norm(result.Sparse - result.tSparse, 2))  # 保存每次实验的稀疏误差

        SumRes.LowrankFrobError.append(np.linalg.norm(result.Lowrank - result.tLowrank,'fro'))  # 保存每次实验的低秩误差
        SumRes.LowrankSpecError.append(np.linalg.norm(result.Lowrank - result.tLowrank, 2))  # 保存每次实验的低秩误差

        if np.linalg.norm(result.tLowrank, 2) < 1e-3:
            # 防止除以非常小的数值
            # 注意: np.linalg.norm(result.tLowrank) 默认为谱范数, 且之前令 norm_factor = 1e-3 是错误的！
            norm_factor = 1
        else:
            norm_factor = np.linalg.norm(result.tLowrank, 2) ** (1 / 2)

        SumRes.TotalFrobError.append( \
            np.linalg.norm(result.tSparse + result.tLowrank - (result.Sparse + result.Lowrank), 'fro') / norm_factor)

        SumRes.TotalSpecError.append( \
            np.linalg.norm(result.tSparse + result.tLowrank - (result.Sparse + result.Lowrank), 2) / norm_factor)

    '''
    注意：result.TrueError[-1][0] 访问的是result.TrueError列表的最后一个元素 [Err1, Err2, Err3 / Scale]的第一个值 Err1,
         代表最后一次迭代的稀疏矩阵误差; result.TrueError[-1][1]; result.TrueError[-1][2] 类似可知.
    '''

    # 计算误差的中位数和标准差，并保存到汇总结果中.
    SparseFrob_median = np.median(SumRes.SparseFrobError)  # 保存所有实验的稀疏误差的中位数和标准差.
    LowrankFrob_median = np.median(SumRes.LowrankFrobError)  # 保存所有实验的低秩误差的中位数和标准差.
    TotalForb_median = np.median(SumRes.TotalFrobError) # 保存所有实验的总误差的中位数和标准差.

    SparseSpec_median = np.median(SumRes.SparseSpecError)  # 保存所有实验的稀疏误差的中位数和标准差.
    LowrankSpec_median = np.median(SumRes.LowrankSpecError)  # 保存所有实验的低秩误差的中位数和标准差.
    TotalSpec_median= np.median(SumRes.TotalSpecError)  # 保存所有实验的总误差的中位数和标准差.

    TrueR_Recovery_median = np.median(SumRes.TrueR_Recovery)  # 保存所有实验中正确选择秩的平均数.分母是迭代收敛的实验次数,分子是迭代收敛且能正确选择秩的实验次数.
    TrueR1_Recovery_median = np.median(SumRes.TrueR1_Recovery)  # 保存所有实验中正确恢复正惯性指数的平均数.分母是迭代收敛的实验次数.

    CostTime_median = np.median(SumRes.CostTime)  # 保存所有实验的耗时的平均数.分母是迭代收敛的实验次数.

    TPR_median = np.median(SumRes.TPR)
    FPR_median = np.median(SumRes.FPR)

    # 调用函数保存结果到 summary_results.txt
    save_summary_to_file(SumRes, Control_Idm, Trial_Idm, d, r, n)

    return SumRes  # 返回包含所有实验统计信息和最优实验结果的 SumRes 对象





#将实验结果保存到文件 Summary_Static_Results.txt
def save_summary_to_file(SumRes, Control_Idm, Trial_Idm, d, r, n, filename="Summary_Dynamic_Results.txt"):
    with open(filename, 'w') as file:  # 使用写入模式，确保每次调用都覆盖之前的文件内容

        # 添加打印控制组、实验组、观测向量维度、隐变量个数、观测向量个数的信息
        file.write(
            f"状态: Dynamic / "
            f"控制组: {Control_Idm} - 实验组: {Trial_Idm} / 观测向量维度: {d} / 隐变量个数: {r} / 观测向量个数: {n}\n")

        # 写入 Point 信息
        file.write("\n=============== Point ===============\n")
        for i, point in enumerate(SumRes.Point):
            file.write(f"Point {i + 1:<3} = {point:<8.6f}\n")

        # 写入 ChooseParamsComb 信息
        file.write("\n=============== ChooseParamsComb ===============\n")
        file.write(f"{'Point':<10}{'alphahat':<10}{'betahat':<10}{'shatprob':<10}{'rhat':<10}\n")
        for i, param_comb in enumerate(SumRes.ChooseParamsComb):
            file.write(
                f"{SumRes.Point[i]:<8.6f}{param_comb[0]:<8.2f}{param_comb[1]:<8.2f}{param_comb[2]:<8.2f}{param_comb[3]:<8.2f}\n")

        # 写入 IterCount 信息
        file.write("\n=============== Dynamic IterCount ===============\n")
        for i, iter_count in enumerate(SumRes.IterCount):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: Iteration {iter_count:<10}\n")

        # 写入 CostTime 信息
        file.write("\n=============== Dynamic CostTime ===============\n")
        for i, cost_time in enumerate(SumRes.CostTime):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {cost_time:<8.6f} seconds\n")

        '''
             Dynamic TrueR1_Recovery
        '''

        # 写入 TrueR1_Recovery 信息
        file.write("\n=============== Dynamic TrueR1_Recovery ===============\n")
        for i, true_r1 in enumerate(SumRes.TrueR1_Recovery):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:8.6f}: {'Recovered' if true_r1 else 'Not Recovered':<10}\n")

        # 计算 TrueR1_Recovery 的总数
        TrueR1_Recovery_count = sum(SumRes.TrueR1_Recovery)

        # 写入 TrueR1_Recovery_count 信息，添加抬头
        file.write("\n=============== Dynamic TrueR1_Recovery Count ===============\n")
        file.write(f"Dynamic TrueR1_Recovery Count = {TrueR1_Recovery_count:<6}\n")

        '''
            Dynamic TrueR_Recovery
        '''

        # 写入 TrueR_Recovery 信息
        file.write("\n=============== Dynamic TrueR_Recovery ===============\n")
        for i, true_r in enumerate(SumRes.TrueR_Recovery):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {'Recovered' if true_r else 'Not Recovered':<10}\n")

        # 计算 TrueR_Recovery 的总数
        TrueR_Recovery_count = sum(SumRes.TrueR_Recovery)

        # 写入 TrueR_Recovery_count 信息，添加抬头
        file.write("\n=============== Dynamic TrueR_Recovery Count ===============\n")
        file.write(f"Dynamic TrueR_Recovery Count = {TrueR_Recovery_count:<6}\n")

        '''
            Dynamic TPR
        '''

        # 写入 TPR 信息
        file.write("\n=============== Dynamic TPR ===============\n")
        for i, tpr in enumerate(SumRes.TPR):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {tpr:<8.6f}\n")

        # 计算 TPR 的中位数和标准差
        TPR_median = np.median(SumRes.TPR)
        TPR_std = np.std(SumRes.TPR)

        # 写入 TPR 中位数和标准差信息，添加抬头
        file.write("\n=============== Dynamic TPR Median & Standard Deviation ===============\n")
        file.write(f"Dynamic TPR Median = {TPR_median:<8.6f}\n")
        file.write(f"Dynamic TPR Standard Deviation = {TPR_std:<8.6f}\n")

        '''
             Dynamic FPR
        '''

        # 写入 FPR 信息
        file.write("\n=============== Dynamic FPR ===============\n")
        for i, fpr in enumerate(SumRes.FPR):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {fpr:<8.6f}\n")

        # 计算 FPR 的中位数和标准差
        FPR_median = np.median(SumRes.FPR)
        FPR_std = np.std(SumRes.FPR)

        # 写入 FPR 中位数和标准差信息，添加抬头
        file.write("\n=============== Dynamic FPR Median & Standard Deviation ===============\n")
        file.write(f"Dynamic FPR Median = {FPR_median:<8.6f}\n")
        file.write(f"Dynamic FPR Standard Deviation = {FPR_std:<8.6f}\n")

        '''
            Dynamic SparseFrobError
        '''

        # 写入 SparseFrobError 信息
        file.write("\n=============== Dynamic SparseFrobError ===============\n")
        for i, s_frob_error in enumerate(SumRes.SparseFrobError):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {s_frob_error:<8.6f}\n")

        # 计算 SparseFrobError 的中位数和标准差
        SparseFrobError_median = np.median(SumRes.SparseFrobError)
        SparseFrobError_std = np.std(SumRes.SparseFrobError)

        # 写入 SparseFrobError 的中位数和标准差信息
        file.write("\n=============== Dynamic SparseFrobError Median & Standard Deviation ===============\n")
        file.write(f"Dynamic SparseFrobError Median = {SparseFrobError_median:<8.6f}\n")
        file.write(f"Dynamic SparseFrobError Standard Deviation = {SparseFrobError_std:<8.6f}\n")

        '''
             Dynamic LowrankFrobError
        '''

        # 写入 LowrankFrobError 信息
        file.write("\n=============== Dynamic LowrankFrobError ===============\n")
        for i, l_frob_error in enumerate(SumRes.LowrankFrobError):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {l_frob_error:<8.6f}\n")

        # 计算 LowrankFrobError 的中位数和标准差
        LowrankFrobError_median = np.median(SumRes.LowrankFrobError)
        LowrankFrobError_std = np.std(SumRes.LowrankFrobError)

        # 写入 LowrankFrobError 的中位数和标准差信息
        file.write("\n=============== Dynamic LowrankFrobError Median and Standard Deviation ===============\n")
        file.write(f"Dynamic LowrankFrobError Median = {LowrankFrobError_median:<8.6f}\n")
        file.write(f"Dynamic LowrankFrobError Standard Deviation = {LowrankFrobError_std:<8.6f}\n")

        '''
             Dynamic TotalFrobError
        '''

        # 写入 TotalFrobError 信息
        file.write("\n=============== Dynamic TotalFrobError ===============\n")
        for i, t_frob_error in enumerate(SumRes.TotalFrobError):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {t_frob_error:<8.6f}\n")

        # 计算 TotalFrobError 的中位数和标准差
        TotalFrobError_median = np.median(SumRes.TotalFrobError)
        TotalFrobError_std = np.std(SumRes.TotalFrobError)

        # 写入 TotalFrobError 的中位数和标准差信息
        file.write("\n=============== Dynamic TotalFrobError Median & Standard Deviation ===============\n")
        file.write(f"Dynamic TotalFrobError Median = {TotalFrobError_median:<8.6f}\n")
        file.write(f"Dynamic TotalFrobError Standard Deviation = {TotalFrobError_std:<8.6f}\n")

        '''
             Dynamic SparseSpecError
        '''

        # 写入 SparseSpecError 信息
        file.write("\n=============== Dynamic SparseSpecError ===============\n")
        for i, s_spec_error in enumerate(SumRes.SparseSpecError):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {s_spec_error:<8.6f}\n")

        # 计算 SparseSpecError 的中位数和标准差
        SparseSpecError_median = np.median(SumRes.SparseSpecError)
        SparseSpecError_std = np.std(SumRes.SparseSpecError)

        # 写入 SparseSpecError 的中位数和标准差信息
        file.write("\n=============== Dynamic SparseSpecError Median & Standard Deviation ===============\n")
        file.write(f"Dynamic SparseSpecError Median = {SparseSpecError_median:<8.6f}\n")
        file.write(f"Dynamic SparseSpecError Standard Deviation = {SparseSpecError_std:<8.6f}\n")

        '''
             Dynamic LowrankSpecError
        '''

        # 写入 LowrankSpecError 信息
        file.write("\n=============== Dynamic LowrankSpecError ===============\n")
        for i, l_spec_error in enumerate(SumRes.LowrankSpecError):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {l_spec_error:<8.6f}\n")

        # 计算 LowrankSpecError 的中位数和标准差
        LowrankSpecError_median = np.median(SumRes.LowrankSpecError)
        LowrankSpecError_std = np.std(SumRes.LowrankSpecError)

        # 写入 LowrankSpecError 的中位数和标准差信息
        file.write("\n=============== Dynamic LowrankSpecError Median & Standard Deviation ===============\n")
        file.write(f"Dynamic LowrankSpecError Median = {LowrankSpecError_median:<8.6f}\n")
        file.write(f"Dynamic LowrankSpecError Standard Deviation = {LowrankSpecError_std:<8.6f}\n")

        '''
              Dynamic TotalSpecError
        '''

        # 写入 TotalSpecError 信息
        file.write("\n=============== Dynamic TotalSpecError ===============\n")
        for i, t_spec_error in enumerate(SumRes.TotalSpecError):
            point = SumRes.Point[i]  # 获取对应的 Point
            file.write(f"Point {i + 1:<3} = {point:<8.6f}: {t_spec_error:<8.6f}\n")

        # 计算 TotalSpecError 的中位数和标准差
        TotalSpecError_median = np.median(SumRes.TotalSpecError)
        TotalSpecError_std = np.std(SumRes.TotalSpecError)

        # 写入 TotalSpecError 的中位数和标准差信息
        file.write("\n=============== Dynamic TotalSpecError Median & Standard Deviation ===============\n")
        file.write(f"Dynamic TotalSpecError Median = {TotalSpecError_median:<8.6f}\n")
        file.write(f"Dynamic TotalSpecError Standard Deviation = {TotalSpecError_std:<8.6f}\n")


    print(f"Summary results have been saved to {filename}")





# 类定义，用于总结模拟结果
class SummarySimuResult:
    def __init__(self):
        self.Point = []
        self.ChooseParamsComb = []
        self.IterCount = []  # 保存每次实验的迭代次数
        self.CostTime = []  # 保存每次实验的运行时间

        self.TrueR1_Recovery = []  # 保存每次实验是否正确恢复正惯性指标的布尔值
        self.TrueR_Recovery = []  # 保存每次实验是否正确选择秩的布尔值

        self.TPR = []  # 保存每次实验的 TPR (True Positive Rate)
        self.FPR = []  # 保存每次实验的 FPR (False Positive Rate)

        self.SparseFrobError = []  # 保存每次实验的迭代中最后一次迭代的稀疏成分与真实稀疏成分的 Frobenius 范数误差
        self.LowrankFrobError = []  # 保存每次实验的迭代中最后一次迭代的低秩成分与真实低秩成分的 Frobenius 范数误差
        self.TotalFrobError = []  # 保存每次实验的迭代中最后一次迭代的总误差的 Frobenius 范数误差

        self.SparseSpecError = []  # 保存每次实验的迭代中最后一次迭代的稀疏成分与真实稀疏成分的谱范数误差
        self.LowrankSpecError = []  # 保存每次实验的迭代中最后一次迭代的低秩成分与真实低秩成分的谱范数误差
        self.TotalSpecError = []  # 保存每次实验的迭代中最后一次迭代的总误差的谱范数误差





SumRes = OurApproach(
    Control_Idm,
    Trial_Idm,
    d,
    r,
    X_combined_vectors,
    Y_combined_vectors,
    X_h2_opt,
    Y_h2_opt,
    global_seed
)
