jizuobiao1 = [17800, 0, 1800]
jizuobiao2 = [12000, 1400, 1400]
jizuobiao3 = [6000,-3000,700]

def segments_all_intersect_sphere_earlystop(p0, p1s, c, r=10, atol=1e-12, include_tangent=True):
    p0 = np.asarray(p0, float)
    c = np.asarray(c, float)
    r = float(r)
    for p1 in np.asarray(p1s, float):
        v = p1 - p0
        w = c - p0
        vv = np.dot(v, v)

        if vv <= atol:
            d0 = np.linalg.norm(p0 - c)
            hit = (d0 <= r + atol) if include_tangent else (d0 < r - atol)
            if not hit:
                return False
            continue
        t = np.clip(np.dot(w, v) / vv, 0.0, 1.0)
        closest = p0 + t * v
        dist = np.linalg.norm(closest - c)
        hit = (dist <= r + atol) if include_tangent else (dist < r - atol)
        if not hit:
            return False
    return True

def objective_function1(solution,t_zhe):
    v,w,t1,t2 = solution
    T = 67
    n = 1000
    dt = T / n
    t_total = 0.0
    for t in np.linspace(0, T, n+1):
        jisudu = v
        t_t = t1
        t_b = t2
        jifangxiang = w
        dzhongxin = daodanzuobiao(m1zuobiao, t)
        yzhongxin = yanwudanzuobiao(jizuobiao1, jifangxiang, jisudu, t_t, t_b, t)
        # print(yzhongxin)
        if segments_all_intersect_sphere_earlystop(
            dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
        ) and t >= t_b and (t<t_zhe[0] or t>t_zhe[1]) :
            t_total += dt
            # print(t)
    return t_total

def objective_function2(solution,t_zhe):
    v,w,t1,t2 = solution
    T = 67
    n = 1000
    dt = T / n
    t_total = 0.0

    for t in np.linspace(0, T, n+1):
        jisudu = v
        t_t = t1
        t_b = t2
        jifangxiang = w
        dzhongxin = daodanzuobiao(m1zuobiao, t)
        yzhongxin = yanwudanzuobiao(jizuobiao2, jifangxiang, jisudu, t_t, t_b, t)
        # print(yzhongxin)
        if segments_all_intersect_sphere_earlystop(
            dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
        ) and t >= t_b and (t<t_zhe[0] or t>t_zhe[1]) :
            t_total += dt
            # print(t)
    return t_total

def objective_function3(solution,t_zhe,t_zhe1):
    v,w,t1,t2 = solution
    T = 67
    n = 1000
    dt = T / n
    t_total = 0.0

    for t in np.linspace(0, T, n+1):
        jisudu = v
        t_t = t1
        t_b = t2
        jifangxiang = w
        dzhongxin = daodanzuobiao(m1zuobiao, t)
        yzhongxin = yanwudanzuobiao(jizuobiao3, jifangxiang, jisudu, t_t, t_b, t)
        # print(yzhongxin)
        if segments_all_intersect_sphere_earlystop(
            dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
        ) and t >= t_b and (t<t_zhe[0] or t>t_zhe[1]) and (t<t_zhe1[0] or t>t_zhe1[1]) :
            t_total += dt
            # print(t)
    return t_total

def constraint_function1(solution):
    v, w, t1, t2 = solution
    return (70 <= v <= 140) and (0 <= t1 <= 14) and (t1 <= t2 <= t1+18) and (0<=(jizuobiao1[0]+v*t2*np.cos(w*pi)) and (jizuobiao1[0]+v*t2*np.cos(w*pi))<=20000) and (-600<=(jizuobiao1[1]+v*t2*np.sin(w*pi)) and (jizuobiao1[1]+v*t2*np.sin(w*pi))<=600)

def constraint_function2(solution):
    v, w, t1, t2 = solution
    return (70 <= v <= 140) and (0 <= t1 <= 27) and (t1 <= t2 <= t1+16) and (1<=w and w<=2) and (0<=(jizuobiao2[0]+v*t2*np.cos(w*pi)) and (jizuobiao2[0]+v*t2*np.cos(w*pi))<=20000) and (-600<=(jizuobiao2[1]+v*t2*np.sin(w*pi)) and (jizuobiao2[1]+v*t2*np.sin(w*pi))<=600)

def constraint_function3(solution):
    v, w, t1, t2 = solution
    return (70 <= v <= 140) and (0 <= t1 <= 46) and (t1 <= t2 <= t1 + 12) and (0<=w and w<=1) and (0<=(jizuobiao3[0]+v*t2*np.cos(w*pi)) and (jizuobiao3[0]+v*t2*np.cos(w*pi))<=20000) and (-600<=(jizuobiao3[1]+v*t2*np.sin(w*pi)) and (jizuobiao3[1]+v*t2*np.sin(w*pi))<=600)
import numpy as np
pi = np.pi

SEED = 10
rng = np.random.default_rng(SEED)

def yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t_c):
    zhongxin = [0, 0, 0]
    if t_c <= t_t:
        zhongxin[0] = jizuobiao[0] + np.cos(jifangxiang * pi) * jisudu * t_c
        zhongxin[1] = jizuobiao[1] + np.sin(jifangxiang * pi) * jisudu * t_c
        zhongxin[2] = jizuobiao[2]
    elif t_c <= t_b:
        zhongxin[0] = jizuobiao[0] + np.cos(jifangxiang * pi) * jisudu * t_c
        zhongxin[1] = jizuobiao[1] + np.sin(jifangxiang * pi) * jisudu * t_c
        zhongxin[2] = jizuobiao[2] - 0.5 * 9.8 * (t_c - t_t) ** 2
    else:
        zhongxin[0] = jizuobiao[0] + np.cos(jifangxiang * pi) * jisudu * t_b
        zhongxin[1] = jizuobiao[1] + np.sin(jifangxiang * pi) * jisudu * t_b
        zhongxin[2] = jizuobiao[2] - 0.5 * 9.8 * (t_b - t_t) ** 2 - 3 * (t_c - t_b)
    return zhongxin

jizuobiao1 = [17800, 0, 1800]
jizuobiao2 = [12000, 1400, 1400]
jizuobiao3 = [6000, -3000, 700]

m1zuobiao = np.array([20000, 0, 2000], dtype=float)
length = np.linalg.norm(m1zuobiao)
m1fangxiang = -m1zuobiao / length
dsudu = 300

def daodanzuobiao(zuobiao, t_c):
    dzhongxin = [0, 0, 0]
    zuobiao = np.array(zuobiao, dtype=float)
    length = np.linalg.norm(zuobiao)
    fangxiang = -zuobiao / length
    dsudu = 300
    dzhongxin[0] = zuobiao[0] + fangxiang[0] * dsudu * t_c
    dzhongxin[1] = zuobiao[1] + fangxiang[1] * dsudu * t_c
    dzhongxin[2] = zuobiao[2] + fangxiang[2] * dsudu * t_c
    return dzhongxin

zhenmubiao = []
x = np.sqrt(49 / (4 + 2 * np.sqrt(2)))
y = (np.sqrt(2) + 1) * x
zhenmubiao.append([x, y + 200, 0])
zhenmubiao.append([y, x + 200, 0])
zhenmubiao.append([y, -x + 200, 0])
zhenmubiao.append([x, -y + 200, 0])
zhenmubiao.append([-x, -y + 200, 0])
zhenmubiao.append([-y, -x + 200, 0])
zhenmubiao.append([-y, x + 200, 0])
zhenmubiao.append([-x, y + 200, 0])
zhenmubiao.append([x, y + 200, 10])
zhenmubiao.append([y, x + 200, 10])
zhenmubiao.append([y, -x + 200, 10])
zhenmubiao.append([x, -y + 200, 10])
zhenmubiao.append([-x, -y + 200, 10])
zhenmubiao.append([-y, -x + 200, 10])
zhenmubiao.append([-y, x + 200, 10])
zhenmubiao.append([-x, y + 200, 10])

def segments_all_intersect_sphere_earlystop(p0, p1s, c, r=10, atol=1e-12, include_tangent=True):
    p0 = np.asarray(p0, float)
    c = np.asarray(c, float)
    r = float(r)
    for p1 in np.asarray(p1s, float):
        v = p1 - p0
        w = c - p0
        vv = np.dot(v, v)
        if vv <= atol:

            d0 = np.linalg.norm(p0 - c)
            hit = (d0 <= r + atol) if include_tangent else (d0 < r - atol)
            if not hit:
                return False
            continue

        t = np.clip(np.dot(w, v) / vv, 0.0, 1.0)
        closest = p0 + t * v
        dist = np.linalg.norm(closest - c)
        hit = (dist <= r + atol) if include_tangent else (dist < r - atol)
        if not hit:
            return False
    return True


def objective_function(solution):
    v, w, t1, t2 = solution
    T = 20
    n = 1000
    dt = T / n

    t_total = 0.0
    for t in np.linspace(0, T, n + 1):
        jisudu = v
        t_t = t1
        t_b = t2
        jifangxiang = w
        dzhongxin = daodanzuobiao(m1zuobiao, t + t_b)
        yzhongxin = yanwudanzuobiao(jizuobiao1, jifangxiang, jisudu, t_t, t_b, t + t_b)
        if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
        ):
            t_total += dt
    return t_total

def constraint_function(solution):
    v, w, t1, t2 = solution
    return (70 <= v <= 140) and (0 <= t1 <= 10) and (t1 <= t2 <= t1 + 12)

V_MIN, V_MAX = 70.0, 140.0
W_MIN, W_MAX = 0, 2.0
T1_MIN, T1_MAX = 0.0, 60.0
T2_MIN, T2_MAX = 0.0, 67.0

LOWER = np.array([V_MIN, W_MIN, T1_MIN, T2_MIN], dtype=float)
UPPER = np.array([V_MAX, W_MAX, T1_MAX, T2_MAX], dtype=float)
SPAN = UPPER - LOWER

def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    if x[3] < x[2]:
        a = x[2]
        x[2] = x[3]
        x[3] = x[2] + 0.8 * (a - x[3])
    if x[3] > T2_MAX:
        x[3] = T2_MAX
        if x[2] > x[3]:
            x[2] = x[3]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 4))
    mask = X[:, 3] < X[:, 2]
    X[mask, 3] = X[mask, 2]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def fitness_of(x):
    s = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
    if not constraint_function(s):
        return -1e12
    return float(objective_function(s))

def pso_optimize(
        swarm_size,
        iters,
        w_start,
        w_end,
        c1,
        c2,
        v_clamp_frac,
        verbose=True
):
    X = random_feasible(swarm_size)  # (N,4)
    V = rng.normal(0.0, 0.05, size=(swarm_size, 4)) * SPAN  # 小速度起步
    VMAX = v_clamp_frac * SPAN

    pbest_pos = X.copy()
    pbest_fit = np.array([fitness_of(x) for x in X], dtype=float)

    g_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    if verbose:
        print(f"[PSO] init: best_fit={gbest_fit:.6f}, best_sol={gbest_pos.tolist()}")

    for it in range(1, iters + 1):
        w_inertia = w_start + (w_end - w_start) * (it / iters)

        r1 = rng.random((swarm_size, 4))
        r2 = rng.random((swarm_size, 4))

        cognitive = c1 * r1 * (pbest_pos - X)
        social = c2 * r2 * (gbest_pos - X)
        V = w_inertia * V + cognitive + social

        V = np.clip(V, -VMAX, VMAX)

        X = X + V
        for i in range(swarm_size):
            X[i] = repair(X[i])

        fits = np.array([fitness_of(x) for x in X], dtype=float)
        improve_mask = fits > pbest_fit
        pbest_pos[improve_mask] = X[improve_mask]
        pbest_fit[improve_mask] = fits[improve_mask]

        g_idx = int(np.argmax(pbest_fit))
        if pbest_fit[g_idx] > gbest_fit:
            gbest_fit = float(pbest_fit[g_idx])
            gbest_pos = pbest_pos[g_idx].copy()

        if verbose and (it % 10 == 0 or it == 1):
            print(f"[PSO] iter={it:03d}  best_fit={gbest_fit:.6f}  best_sol={gbest_pos.tolist()}  w={w_inertia:.3f}")

    return gbest_pos, gbest_fit

pso_best_sol, pso_best_cost = pso_optimize(
    swarm_size=80,
    iters=160,
    w_start=1.2,
    w_end=0.4,
    c1=1.9,
    c2=1.6,
    v_clamp_frac=0.5,
    verbose=True
)

print("\n[PSO] 最优解：", pso_best_sol.tolist())  # [v, w, t1, t2]
print("[PSO] 最大遮挡时间：", pso_best_cost)

li = []
T = 67   # 总时间
n = 1000
t_zhe = [0,0]
for t in np.linspace(0, T, n+1):
    jisudu = pso_best_sol.tolist()[0]
    t_t = pso_best_sol.tolist()[2]
    t_b = pso_best_sol.tolist()[3]
    jifangxiang = pso_best_sol.tolist()[1]
    dzhongxin = daodanzuobiao(m1zuobiao, t)
    yzhongxin = yanwudanzuobiao(jizuobiao1, jifangxiang, jisudu, t_t, t_b, t)
    # print(yzhongxin)
    if segments_all_intersect_sphere_earlystop(
        dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
    ) and t >= t_b:
        li.append(t)
t_zhe[0] = li[0]
t_zhe[1] = li[-1]
t_zhe
SEED = 10
rng = np.random.default_rng(SEED)

V_MIN, V_MAX = 70.0, 140.0
W_MIN, W_MAX = 0,  2.0
T1_MIN, T1_MAX = 0.0, 60.0
T2_MIN, T2_MAX = 0.0, 67.0
# t_zhe = [-1,100]
LOWER = np.array([V_MIN, W_MIN, T1_MIN, T2_MIN], dtype=float)
UPPER = np.array([V_MAX, W_MAX, T1_MAX, T2_MAX], dtype=float)
SPAN  = UPPER - LOWER

def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    if x[3] < x[2]:
        a = x[2]
        x[2] = x[3]
        x[3] = x[2] + 0.8*(a - x[3])
    if x[3] > T2_MAX:
        x[3] = T2_MAX
        if x[2] > x[3]:
            x[2] = x[3]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 4))
    mask = X[:, 3] < X[:, 2]
    X[mask, 3] = X[mask, 2]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def fitness_of(x):
    s = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
    if not constraint_function2(s):
        return -1e12
    return float(objective_function2(s,t_zhe))
def pso_optimize(
    swarm_size,
    iters,
    w_start,
    w_end,
    c1,
    c2,
    v_clamp_frac,
    verbose=True
):
    X = random_feasible(swarm_size)                         # (N,4)
    V = rng.normal(0.0, 0.05, size=(swarm_size, 4)) * SPAN  # 小速度起步
    VMAX = v_clamp_frac * SPAN

    pbest_pos = X.copy()
    pbest_fit = np.array([fitness_of(x) for x in X], dtype=float)

    g_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    if verbose:
        print(f"[PSO] init: best_fit={gbest_fit:.6f}, best_sol={gbest_pos.tolist()}")

    for it in range(1, iters + 1):
        w_inertia = w_start + (w_end - w_start) * (it / iters)

        r1 = rng.random((swarm_size, 4))
        r2 = rng.random((swarm_size, 4))

        cognitive = c1 * r1 * (pbest_pos - X)
        social    = c2 * r2 * (gbest_pos - X)
        V = w_inertia * V + cognitive + social

        V = np.clip(V, -VMAX, VMAX)

        X = X + V
        for i in range(swarm_size):
            X[i] = repair(X[i])

        fits = np.array([fitness_of(x) for x in X], dtype=float)
        improve_mask = fits > pbest_fit
        pbest_pos[improve_mask] = X[improve_mask]
        pbest_fit[improve_mask] = fits[improve_mask]

        g_idx = int(np.argmax(pbest_fit))
        if pbest_fit[g_idx] > gbest_fit:
            gbest_fit = float(pbest_fit[g_idx])
            gbest_pos = pbest_pos[g_idx].copy()

        if verbose and (it % 10 == 0 or it == 1):
            print(f"[PSO] iter={it:03d}  best_fit={gbest_fit:.6f}  best_sol={gbest_pos.tolist()}  w={w_inertia:.3f}")

    return gbest_pos, gbest_fit

pso_best_sol, pso_best_cost = pso_optimize(
    swarm_size=80,
    iters=160,
    w_start=1.2,
    w_end=0.4,
    c1=1.9,
    c2=1.6,
    v_clamp_frac=0.5,
    verbose=True
)

print("\n[PSO] 最优解：", pso_best_sol.tolist())  # [v, w, t1, t2]
print("[PSO] 最大遮挡时间：", pso_best_cost)

yanwudanzuobiao(jizuobiao2,1.3806277988987739,91.98477717801298,8.776390422205623, 15.863576111452169,15.863576111452169)
li = []
t_zhe1 = [0,0]
T = 67
n = 1000
for t in np.linspace(0, T, n + 1):
    jisudu = pso_best_sol.tolist()[0]
    t_t = pso_best_sol.tolist()[2]
    t_b = pso_best_sol.tolist()[3]
    jifangxiang = pso_best_sol.tolist()[1]
    dzhongxin = daodanzuobiao(m1zuobiao, t)
    yzhongxin = yanwudanzuobiao(jizuobiao2, jifangxiang, jisudu, t_t, t_b, t)
    # print(yzhongxin)
    if segments_all_intersect_sphere_earlystop(
            dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
    ) and t >= t_b:
        li.append(t)
t_zhe1[0] = li[0]
t_zhe1[1] = li[-1]
li

SEED = 16
rng = np.random.default_rng(SEED)

V_MIN, V_MAX = 112.0, 113.0
W_MIN, W_MAX = 0,  1.0
T1_MIN, T1_MAX = 0.0, 60.0
T2_MIN, T2_MAX = 0.0, 67.0
# t_zhe = [-1,100]
LOWER = np.array([V_MIN, W_MIN, T1_MIN, T2_MIN], dtype=float)
UPPER = np.array([V_MAX, W_MAX, T1_MAX, T2_MAX], dtype=float)
SPAN  = UPPER - LOWER

def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    if x[3] < x[2]:
        a = x[2]
        x[2] = x[3]
        x[3] = x[2] + 0.8*(a - x[3])
    if x[3] > T2_MAX:
        x[3] = T2_MAX
        if x[2] > x[3]:
            x[2] = x[3]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 4))
    mask = X[:, 3] < X[:, 2]
    X[mask, 3] = X[mask, 2]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def fitness_of(x):
    s = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
    if not constraint_function3(s):
        return -1e12
    return float(objective_function3(s,t_zhe,t_zhe1))
def pso_optimize(
    swarm_size,
    iters,
    w_start,
    w_end,
    c1,
    c2,
    v_clamp_frac,
    verbose=True
):
    X = random_feasible(swarm_size)
    V = rng.normal(0.0, 0.05, size=(swarm_size, 4)) * SPAN
    VMAX = v_clamp_frac * SPAN

    pbest_pos = X.copy()
    pbest_fit = np.array([fitness_of(x) for x in X], dtype=float)

    g_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    if verbose:
        print(f"[PSO] init: best_fit={gbest_fit:.6f}, best_sol={gbest_pos.tolist()}")

    for it in range(1, iters + 1):

        w_inertia = w_start + (w_end - w_start) * (it / iters)

        r1 = rng.random((swarm_size, 4))
        r2 = rng.random((swarm_size, 4))

        cognitive = c1 * r1 * (pbest_pos - X)
        social    = c2 * r2 * (gbest_pos - X)
        V = w_inertia * V + cognitive + social

        V = np.clip(V, -VMAX, VMAX)

        X = X + V
        for i in range(swarm_size):
            X[i] = repair(X[i])

        fits = np.array([fitness_of(x) for x in X], dtype=float)
        improve_mask = fits > pbest_fit
        pbest_pos[improve_mask] = X[improve_mask]
        pbest_fit[improve_mask] = fits[improve_mask]

        g_idx = int(np.argmax(pbest_fit))
        if pbest_fit[g_idx] > gbest_fit:
            gbest_fit = float(pbest_fit[g_idx])
            gbest_pos = pbest_pos[g_idx].copy()

        if verbose and (it % 10 == 0 or it == 1):
            print(f"[PSO] iter={it:03d}  best_fit={gbest_fit:.6f}  best_sol={gbest_pos.tolist()}  w={w_inertia:.3f}")

    return gbest_pos, gbest_fit

pso_best_sol, pso_best_cost = pso_optimize(
    swarm_size=80,
    iters=200,
    w_start=1.8,
    w_end=0.4,
    c1=1.9,
    c2=1.6,
    v_clamp_frac=0.3,
    verbose=True
)

print("\n[PSO] 最优解：", pso_best_sol.tolist())  # [v, w, t1, t2]
print("[PSO] 最大遮挡时间：", pso_best_cost)

#%%
yanwudanzuobiao(jizuobiao3,0.41523234462497244,112.39852407314517,27.037902503723384, 28.370662655537426,28.370662655537426)
#%%
li = []
t_zhe1 = [0,0]
T = 67  # 总时间
n = 1000
for t in np.linspace(0, T, n + 1):
    jisudu = pso_best_sol.tolist()[0]
    t_t = pso_best_sol.tolist()[2]
    t_b = pso_best_sol.tolist()[3]
    jifangxiang = pso_best_sol.tolist()[1]
    dzhongxin = daodanzuobiao(m1zuobiao, t)
    yzhongxin = yanwudanzuobiao(jizuobiao3, jifangxiang, jisudu, t_t, t_b, t)
    # print(yzhongxin)
    if segments_all_intersect_sphere_earlystop(
            dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
    ) and t >= t_b:
        li.append(t)
t_zhe1[0] = li[0]
t_zhe1[1] = li[-1]
li