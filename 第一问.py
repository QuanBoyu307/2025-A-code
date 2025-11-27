#%%
import numpy as np
#%%
def yanwudanzuobiao(jizuobiao,jifangxiang,jisudu,t_t,t_b,t_c):
    zhongxin = [0,0,0]
    if t_c<=t_t:
        zhongxin[0] = jizuobiao[0]+jifangxiang[0]*jisudu*t_c
        zhongxin[1] = jizuobiao[1]+jifangxiang[1]*jisudu*t_c
        zhongxin[2] = jizuobiao[2]+jifangxiang[2]*jisudu*t_c
    elif t_c>t_t and t_c<=t_b:
        zhongxin[0] = jizuobiao[0]+jifangxiang[0]*jisudu*t_c
        zhongxin[1] = jizuobiao[1]+jifangxiang[1]*jisudu*t_c
        zhongxin[2] = jizuobiao[2]+jifangxiang[2]*jisudu*t_t + jifangxiang[2]*jisudu*(t_c-t_t) - 0.5*9.8*(t_c-t_t)**2
    else:
        zhongxin[0] = jizuobiao[0]+jifangxiang[0]*jisudu*t_b
        zhongxin[1] = jizuobiao[1]+jifangxiang[1]*jisudu*t_b
        zhongxin[2] = jizuobiao[2]+jifangxiang[2]*jisudu*t_t + jifangxiang[2]*jisudu*(t_b-t_t) - 0.5*9.8*(t_b-t_t)**2 - 3*(t_c-t_b)
    return zhongxin

jizuobiao = np.array([17800, 0, 1800], dtype=float)

jizuobiao_z = jizuobiao
jizuobiao_z[2] = 0
length = np.linalg.norm(jizuobiao_z)

jifangxiang = -jizuobiao_z / length
jizuobiao = np.array([17800, 0, 1800], dtype=float)
#%%
yanwudanzuobiao(jizuobiao,jifangxiang,120,1.5,5.1,4)
#%%
yanwudanzuobiao(jizuobiao,jifangxiang,120,1.5,5.1,1.5)
#%%
yanwudanzuobiao(jizuobiao,jifangxiang,120,1.5,5.1,5.1)
#%%
m1zuobiao = np.array([20000, 0, 2000], dtype=float)

length = np.linalg.norm(m1zuobiao)

m1fangxiang = -m1zuobiao / length
dsudu = 300
#%%
def daodanzuobiao(zuobiao,t_c):
    dzhongxin = [0,0,0]
    zuobiao = np.array(zuobiao, dtype=float)
    length = np.linalg.norm(zuobiao)
    fangxiang = -zuobiao / length
    dsudu = 300
    dzhongxin[0] = zuobiao[0] + fangxiang[0]*dsudu*t_c
    dzhongxin[1] = zuobiao[1] + fangxiang[1]*dsudu*t_c
    dzhongxin[2] = zuobiao[2] + fangxiang[2]*dsudu*t_c
    return dzhongxin
#%%
daodanzuobiao(m1zuobiao,5.1)
#%%

#%%
zhenmubiao = []
x = np.sqrt(49/(4+2*np.sqrt(2)))
y = (np.sqrt(2)+1)*x
zhenmubiao.append([x,y+200,0])
zhenmubiao.append([y,x+200,0])
zhenmubiao.append([y,-x+200,0])
zhenmubiao.append([x,-y+200,0])
zhenmubiao.append([-x,-y+200,0])
zhenmubiao.append([-y,-x+200,0])
zhenmubiao.append([-y,x+200,0])
zhenmubiao.append([-x,y+200,0])
zhenmubiao.append([x,y+200,10])
zhenmubiao.append([y,x+200,10])
zhenmubiao.append([y,-x+200,10])
zhenmubiao.append([x,-y+200,10])
zhenmubiao.append([-x,-y+200,10])
zhenmubiao.append([-y,-x+200,10])
zhenmubiao.append([-y,x+200,10])
zhenmubiao.append([-x,y+200,10])
#%%
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
            if not hit:                return False
            continue
        t = np.clip(np.dot(w, v) / vv, 0.0, 1.0)
        closest = p0 + t * v
        dist = np.linalg.norm(closest - c)
        hit = (dist <= r + atol) if include_tangent else (dist < r - atol)
        if not hit:
            return False
    return True

#%%
import numpy as np

T = 20  
n = 10000    
dt = T / n

t_total = 0.0

for t in np.linspace(0, T, n+1):
    jisudu = 120
    t_t = 1.5
    t_b = 5.1
    dzhongxin = daodanzuobiao(m1zuobiao, t+t_b)
    yzhongxin = yanwudanzuobiao(jizuobiao, jifangxiang, jisudu, t_t, t_b, t+t_b)
    # print(yzhongxin)
    if segments_all_intersect_sphere_earlystop(
        dzhongxin, zhenmubiao, yzhongxin, r=10, atol=1e-12, include_tangent=True
    ):
        t_total += dt
print("满足条件的总时间:", t_total)
