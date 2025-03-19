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


import math
from joblib import Parallel, delayed

"""
                        =======================================================================
                            Module A : 构造生成数据的模型(实验组和控制组)以及计算相关结果的辅助函数     
                        =======================================================================
"""

# 核函数
def K_h(x, h):
    return np.exp(-0.5 * (x / h) ** 2) / (np.sqrt(2 * np.pi) * h)

# 生成稀疏矩阵 Omega_OO
def sparsemtx(Idm, d, u, rng):
    if Idm == 0:
        matrix = 1 * np.eye(d)
        diag_indices = np.arange(d - 1)
        matrix[diag_indices, diag_indices + 1] = 0.3 * np.sin(np.pi * u / 4) + 0.65
        matrix[diag_indices + 1, diag_indices] = 0.3 * np.sin(np.pi * u / 4) + 0.65
        diag_indices = np.arange(d - 2)
        matrix[diag_indices, diag_indices + 2] = 0.3 * np.cos(np.pi * u / 4) + 0.65
        matrix[diag_indices + 2, diag_indices] = 0.3 * np.cos(np.pi * u / 4) + 0.65
    elif Idm == 1:
        matrix = 1 * np.eye(d)
        value = 1 * (1 + u) ** 2  + 0.38
        indices = [(10 * (k - 1), slice(10 * k - 7, 10 * k)) for k in range(1, d // 10 + 1)]
        for idx, slc in indices:
            matrix[idx, slc] = value
            matrix[slc, idx] = value
    elif Idm == 2:
        matrix = (1.15 + u ** 2) * np.eye(d)
        for i in range(d - 4):
            for j in range(i + 1, i + 4):
                if j < d:
                    value = 0.6 * rng.binomial(1, 0.1) + (1.25 + np.abs(u)) * np.sinc(u)
                    matrix[i, j] = value
                    matrix[j, i] = value
    elif Idm == 3:
        matrix = (2.825 - u ** 2) * np.eye(d)
        for k in range(1, d // 2 + 1):
            value = rng.binomial(1, 0.5) * np.sin(2 * np.pi * u)
            start_idx = 2 * (k - 1)
            end_idx = min(2 * k + 1, d)
            matrix[start_idx, start_idx + 1:end_idx] = value
            matrix[start_idx + 1:end_idx, start_idx] = value
    elif Idm == 4:
        matrix = 1 * np.eye(d)
        epsilon = 1e-12
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    matrix[i, j] = 2
                elif abs(i - j) == 1 and -0.5 <= u <= 1:
                    term = 1 * (0.75 ** 2 - (u - 0.25) ** 2)
                    value = np.exp(u / 2) * np.exp(- (u - 0.25) ** 2 / term) if term > epsilon else 0
                    matrix[i, j] = matrix[j, i] = value
                elif abs(i - j) == 2 and 0.1 <= u <= 1:
                    term = 1 * (0.45 ** 2 - (u - 0.55) ** 2)
                    value = np.exp(u / 2) * np.exp(- (u - 0.55) ** 2 / term) if term > epsilon else 0
                    matrix[i, j] = matrix[j, i] = value
                elif abs(i - j) == 3 and 0.7 <= u <= 1:
                    term = 1 * (0.15 ** 2 - (u - 0.85) ** 2)
                    value = np.exp(u / 2) * np.exp(- (u - 0.85) ** 2 / term) if term > epsilon else 0
                    matrix[i, j] = matrix[j, i] = value
                else:
                    value = 0.125 * rng.binomial(1, 0.1)
                    matrix[i, j] = matrix[j, i] = value
    return matrix

# 生成低秩矩阵 Omega_OH 和 Omega_HO
def lowrankmtx(Idm, d, r, u, rng):
    if Idm == 0:
        Omega_OH = np.full((d, r), 0.3)
        mask = rng.random((d, r)) < 0.1
        Omega_OH[mask] = 0
        Omega_HO = Omega_OH.T
    elif Idm == 1:
        Omega_OH = 0.8 - 0.6 * (rng.uniform(-1.875, 1.875, (d, r))) ** 2
        mask = rng.random((d, r)) < 0.1
        Omega_OH[mask] = 0
        Omega_HO = Omega_OH.T
    elif Idm == 2:
        Omega_OH = rng.uniform(-7 / 10, 7 / 10, (d, r))
        mask = rng.random((d, r)) < 0.1
        Omega_OH[mask] = 0
        Omega_HO = Omega_OH.T
    elif Idm == 3:
        Omega_OH = rng.uniform(-17 / 20, 17 / 20, (d, r))
        mask = rng.random((d, r)) < 0.1
        Omega_OH[mask] = 0
        Omega_HO = Omega_OH.T
    elif Idm == 4:
        Omega_OH = rng.uniform(-1, 1, (d, r))
        mask = rng.random((d, r)) < 0.1
        Omega_OH[mask] = 0
        Omega_HO = Omega_OH.T
    return Omega_OH, Omega_HO

# 生成对角矩阵 D
def diagmtx(Idm, d, r, u, rng):
    if Idm == 0:
        D_1_values = rng.uniform(1, 2, d)
        D_2_values = rng.uniform(1 / 2, 1, r)
    elif Idm == 1:
        D_1_values = rng.uniform(1 / 5, 1 / 2, d)
        D_2_values = rng.uniform(29 / 72, 13 / 16, r)
    elif Idm == 2:
        D_1_values = rng.uniform(1 / 5.5, 1 / 2.5, d)
        D_2_values = rng.uniform(11 / 36 ,5 / 8, r)
    elif Idm == 3:
        D_1_values = rng.uniform(1 / 6, 1 / 3, d)
        D_2_values = rng.uniform(5 / 24 ,7 / 16, r)
    elif Idm == 4:
        D_1_values = rng.uniform(1/ 6.5, 1 / 3.5, d)
        D_2_values = rng.uniform(1 / 9, 1 / 4, r)
    # 构建完整的对角矩阵 D
    D = np.diag(np.concatenate([D_1_values, D_2_values]))
    # 分别提取 D_1 和 D_2
    D_1 = np.diag(D_1_values)
    D_2 = np.diag(D_2_values)
    return D, D_1, D_2

# 计算最小特征值的绝对值
def iota(M):
    return np.abs(np.min(np.linalg.eigvals(M)))


# 生成动态联合精度矩阵
def dynamic_joint_precmtx(Idm, d, r, u, rng):
    Omega_OO = sparsemtx(Idm, d, u, rng)
    Omega_OH, Omega_HO = lowrankmtx(Idm, d, r, u, rng)
    D, D_1, D_2 = diagmtx(Idm, d, r, u, rng)

    if Idm == 0:
       Omega_HH = 1 * np.eye(r)  # Fixed
       Omega_upper = np.hstack((Omega_OO, Omega_OH))
       Omega_lower = np.hstack((Omega_HO, Omega_HH))
       Omega = np.vstack((Omega_upper, Omega_lower))
       iota_Idm = iota(Omega)
       part1 = np.sqrt(D_1) @ (Omega_OO + (iota_Idm + 1) * np.eye(d)) @ np.sqrt(D_1)
       part2 = np.sqrt(D_1) @ Omega_OH @ np.sqrt(D_2)
       part3 = np.sqrt(D_2) @ Omega_HO @ np.sqrt(D_1)
       part4 = np.sqrt(D_2) @ (Omega_HH + (iota_Idm + 1) * np.eye(r)) @ np.sqrt(D_2)
    elif Idm == 1:
       Omega_HH = 1 * np.eye(r)  # Fixed
       Omega_upper = np.hstack((Omega_OO, Omega_OH))
       Omega_lower = np.hstack((Omega_HO, Omega_HH))
       Omega = np.vstack((Omega_upper, Omega_lower))
       iota_Idm = iota(Omega)
       part1 = np.sqrt(D_1) @ (Omega_OO + (iota_Idm + 1) * np.eye(d)) @ np.sqrt(D_1)
       part2 = np.sqrt(D_1) @ Omega_OH @ np.sqrt(D_2)
       part3 = np.sqrt(D_2) @ Omega_HO @ np.sqrt(D_1)
       part4 = np.sqrt(D_2) @ (Omega_HH + (iota_Idm + 1) * np.eye(r)) @ np.sqrt(D_2)
    elif Idm == 2:
       Omega_HH = 1 * np.eye(r)  # Fixed
       Omega_upper = np.hstack((Omega_OO, Omega_OH))
       Omega_lower = np.hstack((Omega_HO, Omega_HH))
       Omega = np.vstack((Omega_upper, Omega_lower))
       iota_Idm = iota(Omega)
       part1 = np.sqrt(D_1) @ (Omega_OO + (iota_Idm + 1) * np.eye(d)) @ np.sqrt(D_1)
       part2 = np.sqrt(D_1) @ Omega_OH @ np.sqrt(D_2)
       part3 = np.sqrt(D_2) @ Omega_HO @ np.sqrt(D_1)
       part4 = np.sqrt(D_2) @ (Omega_HH + (iota_Idm + 1) * np.eye(r)) @ np.sqrt(D_2)
    elif Idm == 3:
       Omega_HH = np.eye(r)  # Fixed
       Omega_upper = np.hstack((Omega_OO, Omega_OH))
       Omega_lower = np.hstack((Omega_HO, Omega_HH))
       Omega = np.vstack((Omega_upper, Omega_lower))
       iota_Idm = iota(Omega)
       part1 = np.sqrt(D_1) @ (Omega_OO + (iota_Idm + 1) * np.eye(d)) @ np.sqrt(D_1)
       part2 = np.sqrt(D_1) @ Omega_OH @ np.sqrt(D_2)
       part3 = np.sqrt(D_2) @ Omega_HO @ np.sqrt(D_1)
       part4 = np.sqrt(D_2) @ (Omega_HH + (iota_Idm + 1) * np.eye(r)) @ np.sqrt(D_2)
    elif Idm == 4:
       Omega_HH = 1 * np.eye(r)  # Fixed
       Omega_upper = np.hstack((Omega_OO, Omega_OH))
       Omega_lower = np.hstack((Omega_HO, Omega_HH))
       Omega = np.vstack((Omega_upper, Omega_lower))
       iota_Idm = iota(Omega)
       part1 = np.sqrt(D_1) @ (Omega_OO + (iota_Idm + 1) * np.eye(d)) @ np.sqrt(D_1)
       part2 = np.sqrt(D_1) @ Omega_OH @ np.sqrt(D_2)
       part3 = np.sqrt(D_2) @ Omega_HO @ np.sqrt(D_1)
       part4 = np.sqrt(D_2) @ (Omega_HH + (iota_Idm + 1) * np.eye(r)) @ np.sqrt(D_2)

    return part1, part2, part3, part4




# 计算动态观察分量的(边际)协方差矩阵 Σoo(u)
def dynamic_obs_covarmtx(Idm, d, r, u, rng):
    part1, part2, part3, part4 = dynamic_joint_precmtx(Idm, d, r, u, rng)

    if Idm == 0:
         sparse_part =  1 * part1
         lowrank_part = - 1 * part2 @ np.linalg.inv(part4) @ part3


    elif Idm == 1:
        sparse_part = 1 * part1
        lowrank_part = -1 * part2 @ np.linalg.inv(part4) @ part3

    elif Idm == 2:
        sparse_part = part1
        lowrank_part = -part2 @ np.linalg.inv(part4) @ part3

    elif Idm == 3:
        sparse_part = 1 * part1
        lowrank_part = -1 * part2 @ np.linalg.inv(part4) @ part3

    elif Idm == 4:
        sparse_part = 1 * part1
        lowrank_part = -1 * part2 @ np.linalg.inv(part4) @ part3


    sigma_oo = np.linalg.inv(sparse_part + lowrank_part)

    return sigma_oo





# 生成单个观测向量的函数, 使用独立的随机数生成器
def generate_single_obs_vector(Idm, d, r, u, rng):
    # 计算动态观测协方差矩阵 Σoo(u)
    sigma_oo = dynamic_obs_covarmtx(Idm, d, r, u, rng)
    # 使用给定的协方差矩阵从多元正态分布中生成随机向量
    return rng.multivariate_normal(np.zeros(d), sigma_oo)

# 生成 [u_i, Y_i] 序列, 其中 Y_i 为时间序列 u_i 对应的观测向量序列.
# 该方案使用 u 值生成独立的随机数生成器 (Generator),
# 保证每个 u 值对应的观测向量 Y_i 生成时有独立的随机化模式.
def generate_combined_vectors(Idm, d, r, u_values, global_seed):
    combined_vectors = []  # 存储 [u_i, Y_i] 的组合向量
    for u in u_values:
        # 将 u 从 [-1, 1] 映射到 [0, 10^6] 区间
        mapped_u = int((u + 1) * 5e5)
        # 使用 global_seed 和 mapped_u 创建一个独立的随机数生成器
        rng = np.random.default_rng(global_seed + mapped_u)
        # 使用该随机数生成器生成观测向量
        obs_vector = generate_single_obs_vector(Idm, d, r, u, rng)
        # 将 u 和对应的观测向量 obs_vector 合并成一个向量
        combined_vector = np.hstack((u, obs_vector))
        combined_vectors.append(combined_vector)  # 将合并向量添加到结果列表中
    return np.array(combined_vectors)  # 将结果列表转换为 NumPy 数组并返回



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



# 定义动态均值估计器 \hat{m}
def m_hat(u_values, obs_vectors, h):
    u = u_values
    Y = obs_vectors
    n = len(u)
    weights = np.array([[K_h(u_j - u_i, h) for u_j in u] for u_i in u])
    np.fill_diagonal(weights, 0)
    weighted_sums = weights @ Y
    normalization = weights.sum(axis=1, keepdims=True)
    return weighted_sums / normalization

# 定义动态协方差估计器 \hat{Sigma}
def Sigma_hat(u_values, obs_vectors, h):
    u = u_values
    Y = obs_vectors
    n, dim = Y.shape
    weights = np.array([[K_h(u_j - u_i, h) for u_j in u] for u_i in u])
    np.fill_diagonal(weights, 0)
    Sigma = np.zeros((n, dim, dim))
    for i in range(n):
        total_weight = np.sum(weights[i])
        weighted_Y = weights[i][:, np.newaxis] * Y
        weighted_average = np.sum(weighted_Y, axis=0) / total_weight
        weighted_sum_yyT = np.dot(weighted_Y.T, Y) / total_weight
        correction = np.outer(weighted_average, weighted_average)
        Sigma[i] = weighted_sum_yyT - correction
    return Sigma


def MSE(u_values, obs_vectors, h):
    m = m_hat(u_values, obs_vectors, h)
    n, dim = m.shape
    mse = (1 / (n * dim)) * np.sum((obs_vectors - m) ** 2)
    return mse

def objective1(h, u_values, obs_vectors):
    return MSE(u_values, obs_vectors, h)

def select_optimal_bandwidth1(u_values, obs_vectors, h_min, h_max):
    h1_values = np.linspace(h_min, h_max, 30)

    # 并行计算 MSE 值
    mse_values = Parallel(n_jobs=-1)(
        delayed(objective1)(h, u_values, obs_vectors) for h in h1_values)
    min_mse = np.min(mse_values)
    h1_opt = h1_values[np.argmin(mse_values)]
    return h1_opt, min_mse, h1_values, mse_values


def cross_validation(u_values, obs_vectors, h, optimal_h1):
    u = u_values
    Y = obs_vectors
    n, dim = Y.shape
    m = m_hat(u, Y, optimal_h1)  # 计算动态均值估计器
    Sigma = Sigma_hat(u, Y, h)  # 计算动态协方差估计器
    residuals = Y - m  # 计算残差

    cond_threshold = 50 # 条件数阈值
    eig_threshold = 0.05 # 最小特征值阈值

    valid_indices = []
    det_Sigma = []
    for i in range(n):
        cond_number = np.linalg.cond(Sigma[i])
        min_eigval = np.min(np.linalg.eigvals(Sigma[i]))
        if cond_number < cond_threshold and min_eigval > eig_threshold:
            valid_indices.append(i)
            det_Sigma.append(np.linalg.det(Sigma[i]))

    valid_count = len(valid_indices)
    # print(f"Number of valid indices: {valid_count}")

    # 如果有效矩阵数量少于总样本数量的一半, 返回一个很高的CV得分(如无穷大)
    # 以避免选择此带宽 h . 这样可以防止由于有效矩阵数量不足导致的结果不可靠.
    if valid_count < n / 2:
        return np.inf

    valid_indices = np.array(valid_indices)
    inv_Sigma = np.linalg.inv(Sigma[valid_indices])  # 计算有效的逆矩阵
    valid_residuals = residuals[valid_indices]  # 过滤有效残差
    term1 = np.array([valid_residuals[i] @ inv_Sigma[i] @ valid_residuals[i] for i in range(valid_count)])
    term2 = np.log(np.array(det_Sigma))
    cv_score = np.sum(term1 + term2) /  valid_count  # 计算交叉验证得分
    return cv_score





def objective2(h, u_values, obs_vectors, optimal_h1):
    return cross_validation(u_values, obs_vectors, h, optimal_h1)

def select_optimal_bandwidth2(u_values, obs_vectors, h_min, h_max, optimal_h1) :
    h2_values = np.linspace(h_min, h_max, 30)
    # 并行计算 CV 值
    cv_values = Parallel(n_jobs=-1)(
        delayed(objective2)(h, u_values, obs_vectors, optimal_h1) for h in h2_values )
    min_cv = np.min(cv_values)
    h2_opt = h2_values[np.argmin(cv_values)]
    return h2_opt, min_cv, h2_values, cv_values



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




# 函数：分割观测向量集
def split_combined_vectors(combined_vectors, split_counter):

    split_seed = split_counter  # 仅使用 split_counter 生成种子

    rng = np.random.default_rng(split_seed)  # 使用生成的种子初始化随机数生成器
    rng.shuffle(combined_vectors)  # 随机打乱顺序
    half = math.ceil(combined_vectors.shape[0] / 2) # 获取数组的行数并分成两半
    group1 = combined_vectors[:half]  # 前一半
    group2 = combined_vectors[half:]  # 后一半

    split_counter += 1  # 计数器自增

    return group1, group2, split_counter  # 返回分割后的组和更新后的计数器



# Sigmahat_cov 动态函数
def Sigmahat_cov(w, h, v, X):
    m = len(v)  # 获取子集的大小
    weights1 = np.array([K_h(v_i - w, h) for v_i in v])
    S1 = sum(weights1[j] * np.outer(X[j], X[j]) for j in range(m))
    S2 = sum(weights1[j] for j in range(m))
    S3 = sum(weights1[j] * X[j] for j in range(m))
    S4 = np.outer(S3, S3)
    Sigmahatcov = (S1 / S2) - (S4 / (S2**2))
    return Sigmahatcov


# 定义函数生成服从[0,1]均匀分布的 w 值
def generate_w_values(n):
    # 若函数还没有 RNG，则初始化一个；否则复用同一个 RNG
    if not hasattr(generate_w_values, "_rng"):
        # 可以在此处改想要的任意固定种子,如 2025
        generate_w_values._rng = np.random.default_rng(2025)
    # 从同一个 RNG 连续抽取 n 个 [0,1] 上的随机值
    w_values = generate_w_values._rng.uniform(0, 1, n)
    return w_values


# 使用条件数和最小特征值相结合的方式, 统计 Sigmahat_cov 为可逆矩阵的数量, 确保筛选出的矩阵在数值计算中是稳定的.
def count_invertible_matrices(n, group, h) :
    w_values = generate_w_values(n)
    v = group[:, 0]
    X = group[:, 1:]
    invertible_count = 0
    cond_threshold = 50  # 条件数阈值
    eig_threshold = 0.05 # 最小特征值的阈值

    for w in w_values:
        Sigma = Sigmahat_cov(w, h, v, X)
        cond_number = np.linalg.cond(Sigma)

        # 如果条件数超过阈值，跳过最小特征值的计算
        if cond_number > cond_threshold:
            continue

        # 仅在条件数合格时计算最小特征值
        min_eigval = np.min(np.linalg.eigvals(Sigma))
        if min_eigval > eig_threshold:
            invertible_count += 1

    return invertible_count



# 使用条件数和最小特征值相结合的方式, 统计 Sigmahat_cov 为可逆矩阵的数量, 确保筛选出的矩阵在数值计算中是稳定的.
def count_invertible_deltmatrics(n, group1, group2, h1, h2) :
    w_values = generate_w_values(n)
    invertible_count = 0
    cond_threshold = 50  # 条件数阈值
    eig_threshold = 0.05 # 最小特征值的阈值

    for w in w_values:
        Sigmahat1_cov = Sigmahat_cov(w, h1, group1[:, 0], group1[:, 1:])
        Sigmahat2_cov = Sigmahat_cov(w, h2, group2[:, 0], group2[:, 1:])
        Deltahat = np.linalg.inv(Sigmahat1_cov) - np.linalg.inv(Sigmahat2_cov)
        cond_number = np.linalg.cond(Deltahat)
        min_eigval = np.min(np.linalg.eigvals(Deltahat))
        min_abs_eigval = np.min(np.abs(np.linalg.eigvals(Deltahat)))
        print(f"w = {w:.6f} / Condition Number = {cond_number:.6f} / Min Eigenvalue = {min_eigval:.6f} / Min Absolute Eigenvalue = {min_abs_eigval:.6f}")


        if cond_number > cond_threshold:
            continue
        if min_abs_eigval > eig_threshold:
            invertible_count += 1

    return invertible_count


"""
                                   Required Auxiliary Functions C
"""

#  calculate_TPR_with_sign
def calculate_TPR(Sparse, tSparse, threshold=1e-3):
    # 判断元素是否为非零并且符号一致
    # estimated_nonzero_pos 表示在估计矩阵 Sparse 中, 大于正阈值 threshold 的位置, 即正值的位置
    estimated_nonzero_pos = (Sparse > threshold)

    # estimated_nonzero_neg 表示在估计矩阵 Sparse 中, 小于负阈值 -threshold 的位置, 即负值的位置
    estimated_nonzero_neg = (Sparse < -threshold)

    # true_nonzero_pos 表示在真实矩阵 tSparse 中, 大于正阈值 threshold 的位置, 即正值的位置
    true_nonzero_pos = (tSparse > threshold)

    # true_nonzero_neg 表示在真实矩阵 tSparse 中, 小于负阈值 -threshold 的位置, 即负值的位置
    true_nonzero_neg = (tSparse < -threshold)

    # 计算 TP (真实和预测为正数的元素个数) + (真实和预测为负数的元素个数)
    # TP 是 True Positives，表示在同一位置上, 预测和真实矩阵都为正数的元素数量, 或者都为负数的元素数量
    # 通过 & 操作符, 判断在同一位置上是否同时满足条件 (同为正或同为负), 并用 | 操作符将这两种情况合并
    TP = np.sum((estimated_nonzero_pos & true_nonzero_pos) | (estimated_nonzero_neg & true_nonzero_neg))

    # 计算 FN (真实为非零,但预测为零或者符号不一致的元素个数)
    # FN 是 False Negatives, 表示真实矩阵为非零(正数或负数),而估计矩阵为零或符号不一致的元素数量
    # 通过 ~ 操作符取反, 表示位置上预测为零或符号不一致的情况
    FN = np.sum((true_nonzero_pos & ~estimated_nonzero_pos) | (true_nonzero_neg & ~estimated_nonzero_neg))

    # 计算 TPR (True Positive Rate)
    # TPR = TP / (TP + FN) 表示真实为非零元素中, 正确预测为非零的比例
    if TP + FN == 0:
        return 0  # 避免除零错误, 当 TP + FN 为 0 时, 直接返回 0 , 因为没有真实的非零元素
    TPR = TP / (TP + FN)
    return TPR


def calculate_FPR(Sparse, tSparse, threshold=1e-3):
    # true_zero 表示在真实矩阵 tSparse 中, 绝对值小于等于阈值 threshold 的位置
    true_zero = (np.abs(tSparse) <= threshold)

    # estimated_nonzero 表示在估计矩阵 Sparse 中, 绝对值大于阈值 threshold 的位置
    estimated_nonzero = (np.abs(Sparse) > threshold)

    # 计算 FP (真实为零但预测为非零的元素个数)
    FP = np.sum(estimated_nonzero & true_zero)

    # 计算 FPR (False Positive Rate)
    # FPR = FP / true_zero的元素个数
    if np.sum(true_zero) == 0:
        return 0  # 避免除零错误
    FPR = FP / np.sum(true_zero)
    return FPR



'''
==================================================================================================
'''

import numpy as np
import cvxpy as cp

def dynamic_clime(Sigma_hat, theta=0.0125):
    """
    动态CLIME方法求解 \tilde{A}(u):
    minimize ||A||_1 subject to ||\hat{\Sigma}(u) A - I||_\infty <= theta
    并对结果进行对称化处理
    """
    d = Sigma_hat.shape[0]
    A = cp.Variable((d, d))
    I = np.eye(d)

    # 定义目标函数和约束条件
    objective = cp.Minimize(cp.norm(A, 1))
    constraints = [cp.norm(Sigma_hat @ A - I, 'inf') <= theta]

    # 定义优化问题并求解
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # 得到估计的矩阵 \hat{A}_1(u)
    A_hat = A.value

    # 对 \hat{A}_1(u) 进行对称化处理，得到最终估计的逆矩阵 \tilde{A}(u)
    A_sym = np.zeros_like(A_hat)

    for j in range(d):
        for k in range(j, d):  # 只遍历上三角部分
            if abs(A_hat[j, k]) <= abs(A_hat[k, j]):
                A_sym[j, k] = A_hat[j, k]
                A_sym[k, j] = A_hat[j, k]
            else:
                A_sym[j, k] = A_hat[k, j]
                A_sym[k, j] = A_hat[k, j]

    return A_sym