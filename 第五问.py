import numpy as np
pi = np.pi

SEED = 20
rng = np.random.default_rng(SEED)

jizuobiao1 = [17800, 0, 1800]
jizuobiao2 = [12000, 1400, 1400]
jizuobiao3 = [6000, -3000, 700]
jizuobiao4 = [11000, 2000, 1800]
jizuobiao5 = [13000, -2000, 1300]

def yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t_c):

    zhongxin = [0, 0, 0]
    if t_c <= t_t:
        zhongxin[0] = jizuobiao[0] + np.cos(jifangxiang*pi) * jisudu * t_c
        zhongxin[1] = jizuobiao[1] + np.sin(jifangxiang*pi) * jisudu * t_c
        zhongxin[2] = jizuobiao[2]
    elif t_c <= t_b:
        zhongxin[0] = jizuobiao[0] + np.cos(jifangxiang*pi) * jisudu * t_c
        zhongxin[1] = jizuobiao[1] + np.sin(jifangxiang*pi) * jisudu * t_c
        zhongxin[2] = jizuobiao[2] - 0.5*9.8*(t_c - t_t)**2
    else:
        zhongxin[0] = jizuobiao[0] + np.cos(jifangxiang*pi) * jisudu * t_b
        zhongxin[1] = jizuobiao[1] + np.sin(jifangxiang*pi) * jisudu * t_b
        zhongxin[2] = jizuobiao[2] - 0.5*9.8*(t_b - t_t)**2 - 3*(t_c - t_b)
    return zhongxin

m1zuobiao = np.array([20000, 0, 2000], dtype=float)
length = np.linalg.norm(m1zuobiao)
m1fangxiang = -m1zuobiao / length
dsudu = 300
m2zuobiao = np.array([19000, 600, 2100], dtype=float)
length = np.linalg.norm(m2zuobiao)
m2fangxiang = -m2zuobiao / length
m3zuobiao = np.array([18000, -600, 1900], dtype=float)
length = np.linalg.norm(m3zuobiao)
m3fangxiang = -m3zuobiao / length

def daodanzuobiao(zuobiao, t_c):
    dzhongxin = [0, 0, 0]
    zuobiao = np.array(zuobiao, dtype=float)
    length = np.linalg.norm(zuobiao)
    fangxiang = -zuobiao / length
    dsudu = 300
    dzhongxin[0] = zuobiao[0] + fangxiang[0]*dsudu*t_c
    dzhongxin[1] = zuobiao[1] + fangxiang[1]*dsudu*t_c
    dzhongxin[2] = zuobiao[2] + fangxiang[2]*dsudu*t_c
    return dzhongxin

zhenmubiao = []
x = np.sqrt(49/(4+2*np.sqrt(2)))
y = (np.sqrt(2)+1)*x
zhenmubiao.append([x,   y+200,   0])
zhenmubiao.append([y,   x+200,   0])
zhenmubiao.append([y,  -x+200,   0])
zhenmubiao.append([x,  -y+200,   0])
zhenmubiao.append([-x, -y+200,   0])
zhenmubiao.append([-y, -x+200,   0])
zhenmubiao.append([-y,  x+200,   0])
zhenmubiao.append([-x,  y+200,   0])
zhenmubiao.append([x,   y+200,  10])
zhenmubiao.append([y,   x+200,  10])
zhenmubiao.append([y,  -x+200,  10])
zhenmubiao.append([x,  -y+200,  10])
zhenmubiao.append([-x, -y+200,  10])
zhenmubiao.append([-y, -x+200,  10])
zhenmubiao.append([-y,  x+200,  10])
zhenmubiao.append([-x,  y+200,  10])

def segments_all_intersect_sphere_earlystop(p0, p1s, c, r=10, atol=1e-12, include_tangent=True):

    p0 = np.asarray(p0, float)
    c = np.asarray(c, float)
    r = float(r)
    for p1 in np.asarray(p1s, float):
        v = p1 - p0
        w = c  - p0
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

def objective_function1(solution):
    t_total = 0.0
    T1 = 67
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T1, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m1zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao1, jifangxiang, jisudu, t_t, t_b, t)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b:
                t_total += dt
                break

    T2 = 64
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T2, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m2zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao1, jifangxiang, jisudu, t_t, t_b, t)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b:
                t_total += dt
                break
    T3 = 61
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T3, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m3zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao1, jifangxiang, jisudu, t_t, t_b, t)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b:
                t_total += dt
                break

    return t_total
def constraint_function1(solution):
    v, w = solution[0],solution[1]
    t1, t2 = solution[2],solution[3]
    t3, t4 = solution[4],solution[5]
    t5, t6 = solution[6],solution[7]
    return (70 <= v <= 140) and (0 <= t1 <= 14) and (t1 <= t2 <= t1+18) and (t1+1 <= t3 < 14) and (t3 <= t4 <= t3+18) and (t3+1 <= t5 < 14) and (t5 <= t6 <= t5+18)

V_MIN, V_MAX = 70.0, 140.0
W_MIN, W_MAX = 0,  2.0

T_START_MIN, T_START_MAX = 0.0, 14.0
T_END_MIN,   T_END_MAX   = 0.0, 32.0

LOWER = np.array([
    V_MIN, W_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN
], dtype=float)

UPPER = np.array([
    V_MAX, W_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX
], dtype=float)

SPAN = UPPER - LOWER
def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    x[1] = x[1] % 2.0
    # 三对 (t1,t2), (t3,t4), (t5,t6)
    for i in [2, 4, 6]:
        if x[i + 1] < x[i]:
            x[i + 1] = x[i]
        if x[i + 1] > T_END_MAX:
            x[i + 1] = T_END_MAX
            if x[i] > x[i + 1]:
                x[i] = x[i + 1]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i+1] < X[:, i]
        X[mask, i+1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    x[1] = x[1] % 2.0
    for i in [2, 4, 6]:
        if x[i+1] < x[i]:
            a = x[i]
            x[i] = x[i+1]+0.2*(a-x[i+1])
            x[i+1] = a
        if x[i+1] > T_END_MAX:
            x[i+1] = T_START_MAX
            if x[i] > x[i+1]:
                x[i] = x[i+1]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i+1] < X[:, i]
        X[mask, i+1] = X[mask, i]

    for i in range(n):
        X[i] = repair(X[i])
    return X

def fitness_of(x):
    s = [float(val) for val in x]
    if not constraint_function1(s):
        return -1e12
    return float(objective_function1(s))

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
    D = LOWER.size
    X = random_feasible(swarm_size)
    V = rng.normal(0.0, 0.05, size=(swarm_size, D)) * SPAN
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

        r1 = rng.random((swarm_size, D))
        r2 = rng.random((swarm_size, D))

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

print("\n[PSO] 最优解（[v, w, t1, t2, t3, t4, t5, t6]）:", pso_best_sol.tolist())
print("[PSO] 最大收益：", pso_best_cost)
solution1 = pso_best_sol.tolist()
li_zong = [[],[],[]]
pso_best_sol.tolist()
def gengxinhanshu(zuiyoujie,jizuobiao,li_zong):
    T = 67
    n = 1000
    for i in range(3):
        li = []
        rongqi = [-100,100]
        for t in np.linspace(0, T, n+1):
            jisudu = zuiyoujie[0]
            t_t = zuiyoujie[2+i*2]
            t_b = zuiyoujie[3+i*2]
            jifangxiang = zuiyoujie[1]
            dzhongxin = daodanzuobiao(m1zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)
            # print(yzhongxin)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and t <= t_b+20:
                li.append(t)
            # print(li)
        if len(li) >= 2:
            rongqi[0] = li[0]
            rongqi[1] = li[-1]
            li_zong[0].append(rongqi)

    T = 64
    n = 1000
    for i in range(3):
        li = []
        rongqi = [-100,100]
        for t in np.linspace(0, T, n+1):
            jisudu = zuiyoujie[0]
            t_t = zuiyoujie[2+i*2]
            t_b = zuiyoujie[3+i*2]
            jifangxiang = zuiyoujie[1]
            dzhongxin = daodanzuobiao(m2zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and t <= t_b+20:
                li.append(t)
        if len(li) >= 2:
            rongqi[0] = li[0]
            rongqi[1] = li[-1]
            li_zong[1].append(rongqi)

    T = 61
    n = 1000
    for i in range(3):
        li = []
        rongqi = [-100,100]
        for t in np.linspace(0, T, n+1):
            jisudu = zuiyoujie[0]
            t_t = zuiyoujie[2+i*2]
            t_b = zuiyoujie[3+i*2]
            jifangxiang = zuiyoujie[1]
            dzhongxin = daodanzuobiao(m3zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)
            # print(yzhongxin)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and t <= t_b+20:
                li.append(t)
        if len(li) >= 2:
            rongqi[0] = li[0]
            rongqi[1] = li[-1]
            li_zong[2].append(rongqi)
    return li_zong

gengxinhanshu(pso_best_sol.tolist(),jizuobiao1,li_zong)
def objective_function(solution,jizuobiao,li_zong):

    t_total = 0.0
    T1 = 67
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T1, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m1zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)

            intervals = li_zong[0]
            in_interval = any(start <= t <= end for start, end in intervals)

            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and not in_interval and t <= t_b+20:
                t_total += dt
                break

    T2 = 64
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T2, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m2zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)

            intervals = li_zong[1]
            in_interval = any(start <= t <= end for start, end in intervals)

            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and not in_interval and t <= t_b+20:
                t_total += dt
                break
    T3 = 61
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T3, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m3zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)

            intervals = li_zong[2]
            in_interval = any(start <= t <= end for start, end in intervals)

            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and not in_interval and t <= t_b+20:
                t_total += dt
                break
    return t_total
def objective_function2(solution,jizuobiao,li_zong):

    t_total = 0.0
    T1 = 67
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T1, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m1zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)

            intervals = li_zong[0]
            in_interval = any(start <= t <= end for start, end in intervals)

            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and not in_interval and t <= t_b+20:
                t_total += dt*0.5
                break

    T2 = 64
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T2, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m2zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)

            intervals = li_zong[1]
            in_interval = any(start <= t <= end for start, end in intervals)

            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and not in_interval and t <= t_b+20:
                t_total += dt
                break
    T3 = 61
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T3, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m3zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t)

            intervals = li_zong[2]
            in_interval = any(start <= t <= end for start, end in intervals)

            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b and not in_interval and t <= t_b+20:
                t_total += dt
                break
    return t_total
def constraint_function2(solution):
    v, w = solution[0],solution[1]
    t1, t2 = solution[2],solution[3]
    t3, t4 = solution[4],solution[5]
    t5, t6 = solution[6],solution[7]
    return (70 <= v <= 140) and (0 <= t1 <= 67) and (t1 <= t2 <= t1+18) and (t1+1 <= t3 < 67) and (t3 <= t4 <= t3+18) and (t3+1 <= t5 < 67) and (t5 <= t6 <= t5+18) and (1 <= w <= 2) and (0<=(jizuobiao2[0]+v*t2*np.cos(w*pi)) and (jizuobiao2[0]+v*t2*np.cos(w*pi))<=20000) and (-600<=(jizuobiao2[1]+v*t2*np.sin(w*pi)) and (jizuobiao2[1]+v*t2*np.sin(w*pi))<=600)
def constraint_function5(solution):
    v, w = solution[0],solution[1]
    t1, t2 = solution[2],solution[3]
    t3, t4 = solution[4],solution[5]
    t5, t6 = solution[6],solution[7]
    return (70 <= v <= 140) and (0 <= t1 <= 67) and (t1 <= t2 <= t1+18) and (t1+1 <= t3 < 67) and (t3 <= t4 <= t3+18) and (t3+1 <= t5 < 67) and (t5 <= t6 <= t5+18) and (0 <= w <= 1) and (0<=(jizuobiao5[0]+v*t2*np.cos(w*pi)) and (jizuobiao5[0]+v*t2*np.cos(w*pi))<=20000) and (-600<=(jizuobiao5[1]+v*t2*np.sin(w*pi)) and (jizuobiao5[1]+v*t2*np.sin(w*pi))<=600)
def constraint_function4(solution):
    v, w = solution[0],solution[1]
    t1, t2 = solution[2],solution[3]
    t3, t4 = solution[4],solution[5]
    t5, t6 = solution[6],solution[7]
    return (70 <= v <= 140) and (0 <= t1 <= 67) and (t1 <= t2 <= t1+18) and (t1+1 <= t3 < 67) and (t3 <= t4 <= t3+18) and (t3+1 <= t5 < 67) and (t5 <= t6 <= t5+18) and (1 <= w <= 2) and (0<=(jizuobiao4[0]+v*t2*np.cos(w*pi)) and (jizuobiao4[0]+v*t2*np.cos(w*pi))<=20000) and (-600<=(jizuobiao4[1]+v*t2*np.sin(w*pi)) and (jizuobiao4[1]+v*t2*np.sin(w*pi))<=600)
def constraint_function3(solution):
    v, w = solution[0],solution[1]
    t1, t2 = solution[2],solution[3]
    t3, t4 = solution[4],solution[5]
    t5, t6 = solution[6],solution[7]
    return (70 <= v <= 140) and (0 <= t1 <= 67) and (t1 <= t2 <= t1+12) and (t1+1 <= t3 < 67) and (t3 <= t4 <= t3+12) and (t3+1 <= t5 < 67) and (t5 <= t6 <= t5+12) and (0 <= w <= 1) and (0<=(jizuobiao3[0]+v*t2*np.cos(w*pi)) and (jizuobiao3[0]+v*t2*np.cos(w*pi))<=20000) and (-600<=(jizuobiao3[1]+v*t2*np.sin(w*pi)) and (jizuobiao3[1]+v*t2*np.sin(w*pi))<=600)
SEED = 10
rng = np.random.default_rng(SEED)
V_MIN, V_MAX = 70.0, 140.0
W_MIN, W_MAX = 0, 2.0

T_START_MIN, T_START_MAX = 0.0, 14.0
T_END_MIN, T_END_MAX = 0.0, 32.0

LOWER = np.array([
    V_MIN, W_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN
], dtype=float)

UPPER = np.array([
    V_MAX, W_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX
], dtype=float)

SPAN = UPPER - LOWER

def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    x[1] = x[1] % 2.0
    for i in [2, 4, 6]:
        if x[i+1] < x[i]:
            a = x[i]
            x[i] = x[i+1]+0.5*(a-x[i+1])
            x[i+1] = a
        if x[i+1] > T_END_MAX:
            x[i+1] = T_START_MAX
            if x[i] > x[i+1]:
                x[i] = x[i+1]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def fitness_of(x):
    s = [float(val) for val in x]
    if not constraint_function5(s):
        return -1e12
    return float(objective_function(s,jizuobiao5,li_zong))


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
    D = LOWER.size
    X = random_feasible(swarm_size)
    V = rng.normal(0.0, 0.05, size=(swarm_size, D)) * SPAN
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
        r1 = rng.random((swarm_size, D))
        r2 = rng.random((swarm_size, D))

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
    v_clamp_frac=0.8,
    verbose=True
)

print("\n[PSO] 最优解（[v, w, t1, t2, t3, t4, t5, t6]）:", pso_best_sol.tolist())
print("[PSO] 最大收益：", pso_best_cost)

solution5 = pso_best_sol.tolist()

pso_best_sol.tolist()

gengxinhanshu(pso_best_sol.tolist(),jizuobiao5,li_zong)

li_zong5 = li_zong

SEED = 20
rng = np.random.default_rng(SEED)

V_MIN, V_MAX = 70.0, 140.0  # v
W_MIN, W_MAX = 0, 2.0  # w

T_START_MIN, T_START_MAX = 0.0, 14.0
T_END_MIN, T_END_MAX = 0.0, 32.0

LOWER = np.array([
    V_MIN, W_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN
], dtype=float)

UPPER = np.array([
    V_MAX, W_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX
], dtype=float)

SPAN = UPPER - LOWER


def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    x[1] = x[1] % 2.0
    for i in [2, 4, 6]:
        if x[i+1] < x[i]:
            a = x[i]
            x[i] = x[i+1]+0.5*(a-x[i+1])
            x[i+1] = a
        if x[i+1] > T_END_MAX:
            x[i+1] = T_START_MAX
            if x[i] > x[i+1]:
                x[i] = x[i+1]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def fitness_of(x):
    s = [float(val) for val in x]
    if not constraint_function2(s):
        return -1e12
    return float(objective_function2(s,jizuobiao2,li_zong))

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
    D = LOWER.size
    X = random_feasible(swarm_size)
    V = rng.normal(0.0, 0.05, size=(swarm_size, D)) * SPAN
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
        r1 = rng.random((swarm_size, D))
        r2 = rng.random((swarm_size, D))
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

print("\n[PSO] 最优解（[v, w, t1, t2, t3, t4, t5, t6]）:", pso_best_sol.tolist())
print("[PSO] 最大收益：", pso_best_cost)

objective_function(pso_best_sol.tolist(),jizuobiao2,li_zong)

gengxinhanshu(pso_best_sol.tolist(),jizuobiao2,li_zong)

li_zong5

li_zong2 = li_zong

solution2 = pso_best_sol.tolist()

SEED = 23
rng = np.random.default_rng(SEED)

V_MIN, V_MAX = 70.0, 140.0  # v
W_MIN, W_MAX = 0, 2.0  # w

T_START_MIN, T_START_MAX = 0.0, 14.0
T_END_MIN, T_END_MAX = 0.0, 32.0

LOWER = np.array([
    V_MIN, W_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN
], dtype=float)

UPPER = np.array([
    V_MAX, W_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX
], dtype=float)

SPAN = UPPER - LOWER


def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    x[1] = x[1] % 2.0
    for i in [2, 4, 6]:
        if x[i+1] < x[i]:
            a = x[i]
            x[i] = x[i+1]+0.5*(a-x[i+1])
            x[i+1] = a
        if x[i+1] > T_END_MAX:
            x[i+1] = T_START_MAX
            if x[i] > x[i+1]:
                x[i] = x[i+1]
    return x

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def fitness_of(x):
    s = [float(val) for val in x]
    if not constraint_function4(s):
        return -1e12
    return float(objective_function(s,jizuobiao4,li_zong))

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
    D = LOWER.size
    X = random_feasible(swarm_size)
    V = rng.normal(0.0, 0.05, size=(swarm_size, D)) * SPAN
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
        r1 = rng.random((swarm_size, D))
        r2 = rng.random((swarm_size, D))
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

print("\n[PSO] 最优解（[v, w, t1, t2, t3, t4, t5, t6]）:", pso_best_sol.tolist())
print("[PSO] 最大收益：", pso_best_cost)

gengxinhanshu(pso_best_sol.tolist(),jizuobiao4,li_zong)

li_zong4 = li_zong

solution4 = pso_best_sol.tolist()

SEED = 2021
rng = np.random.default_rng(SEED)

V_MIN, V_MAX = 90.0, 120.0
W_MIN, W_MAX = 0, 1.0


T_START_MIN, T_START_MAX = 0.0, 60.0
T_END_MIN, T_END_MAX = 0.0, 67.0

LOWER = np.array([
    V_MIN, W_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN,
    T_START_MIN, T_END_MIN
], dtype=float)

UPPER = np.array([
    V_MAX, W_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX,
    T_START_MAX, T_END_MAX
], dtype=float)

SPAN = UPPER - LOWER


def repair(x):
    x = np.array(x, dtype=float)
    x = np.minimum(np.maximum(x, LOWER), UPPER)
    x[1] = x[1] % 2.0
    for i in [2, 4, 6]:
        if x[i+1] < x[i]:
            a = x[i]
            x[i] = x[i+1]+0.5*(a-x[i+1])
            x[i+1] = a
        if x[i+1] > T_END_MAX:
            x[i+1] = T_START_MAX
            if x[i] > x[i+1]:
                x[i] = x[i+1]
    return x


def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])
    return X

def random_feasible(n):
    X = rng.uniform(LOWER, UPPER, size=(n, 8))
    for i in [2, 4, 6]:
        mask = X[:, i + 1] < X[:, i]
        X[mask, i + 1] = X[mask, i]
    for i in range(n):
        X[i] = repair(X[i])

    return X

def fitness_of(x):
    s = [float(val) for val in x]
    if not constraint_function3(s):
        return -1e12
    return float(objective_function(s,jizuobiao3,li_zong))

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
    D = LOWER.size
    X = random_feasible(swarm_size)
    V = rng.normal(0.0, 0.05, size=(swarm_size, D)) * SPAN
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
        r1 = rng.random((swarm_size, D))
        r2 = rng.random((swarm_size, D))
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
    w_start=1.5,
    w_end=0.4,
    c1=1.9,
    c2=0.5,
    v_clamp_frac=0.05,
    verbose=True
)

print("\n[PSO] 最优解（[v, w, t1, t2, t3, t4, t5, t6]）:", pso_best_sol.tolist())
print("[PSO] 最大收益：", pso_best_cost)

def objective_function_f(solution):
    t_total = [[0,0,0],[0,0,0],[0,0,0]]
    T1 = 67
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T1, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m1zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao5, jifangxiang, jisudu, t_t, t_b, t)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b:
                t_total[i][0] += dt
                break

    T2 = 64
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T2, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m2zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao5, jifangxiang, jisudu, t_t, t_b, t)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b:
                t_total[i][1] += dt
                break
    T3 = 61
    n = 500
    dt = T1 / n
    for t in np.linspace(0, T3, n+1):
        for i in range(3):
            v, w = solution[0],solution[1]
            t1, t2 = solution[2+i*2],solution[3+i*2]
            jisudu = v
            t_t = t1
            t_b = t2
            jifangxiang = w
            dzhongxin = daodanzuobiao(m3zuobiao, t)
            yzhongxin = yanwudanzuobiao(jizuobiao5, jifangxiang, jisudu, t_t, t_b, t)
            if segments_all_intersect_sphere_earlystop(
                dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
            ) and t >= t_b:
                t_total[i][2] += dt
                break

    return t_total

objective_function_f(solution5)

solution3 = pso_best_sol.tolist()

gengxinhanshu(pso_best_sol.tolist(),jizuobiao3,li_zong)

objective_function(solution1,jizuobiao1,[[],[],[]])+objective_function(solution5,jizuobiao5,[[[4.757000000000001, 7.37], [1.072, 5.226]], [], []])+objective_function(solution2,jizuobiao2,[[[4.757000000000001, 7.37], [1.072, 5.226]], [], [[15.494, 19.093]]])+objective_function(solution4,jizuobiao4,[[[4.757000000000001, 7.37], [1.072, 5.226], [16.482, 20.234]],
 [[12.416, 16.256]],
 [[15.494, 19.093]]])+objective_function(solution3,jizuobiao3,[[[4.757000000000001, 7.37], [1.072, 5.226], [16.482, 20.234]],
 [[12.416, 16.256], [22.656, 25.984]],
 [[15.494, 19.093]]])

