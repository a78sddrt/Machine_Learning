import numpy as np
import matplotlib.pyplot as plt
#迷宮的初始狀態

#圖的大小
fig=plt.figure(figsize=(5,5))
ax=plt.gca()

#畫牆壁
plt.plot([1,1],[0,1], color='black', linewidth=2)
plt.plot([1,2],[2,2], color='black', linewidth=2)
plt.plot([2,2],[2,1], color='black', linewidth=2)
plt.plot([2,3],[1,1], color='black', linewidth=2)

#中心文字
plt.text(0.5,2.5,'S0',size=14, ha='center')
plt.text(1.5,2.5,'S1',size=14, ha='center')
plt.text(2.5,2.5,'S2',size=14, ha='center')
plt.text(0.5,1.5,'S3',size=14, ha='center')
plt.text(1.5,1.5,'S4',size=14, ha='center')
plt.text(2.5,1.5,'S5',size=14, ha='center')
plt.text(0.5,0.5,'S6',size=14, ha='center')
plt.text(1.5,0.5,'S7',size=14, ha='center')
plt.text(2.5,0.5,'S8',size=14, ha='center')
plt.text(0.5,0.3,'START', ha='center')
plt.text(2.5,0.3,'GOAL', ha='center')

#繪圖範圍
ax.set_xlim(0,3)
ax.set_ylim(0,3)

#plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

#畫圓
line,=ax.plot([0.5],[2.5],marker="o", color='g', markersize=60)


#可行動方向
theta_0=np.array([[np.nan,1,1,np.nan], #s0
                  [np.nan,1,np.nan,1], #s1
                  [np.nan,np.nan,1,1], #s2
                  [1,1,1,np.nan], #s3
                  [np.nan, np.nan,1,1], #s4
                  [1,np.nan,np.nan,np.nan], #5
                  [1,np.nan,np.nan,np.nan], #6
                  [1,1,np.nan,np.nan], #7
    ])
#建造行動價值表格(一開始不知道，所以先random)
[a,b]=theta_0.shape
Q=np.random.rand(a,b)*theta_0 #乘以theta_0, 將朝向牆壁的值設為nan

#隨機動作策略pi_0
def simple_convert_into_pi_from_theta(theta):
    [m,n]=theta.shape
    pi=np.zeros((m,n))
    for i in range(m):
        pi[i,:]=theta[i,:]/np.nansum(theta[i,:])
        pi=np.nan_to_num(pi)
    return pi
    
pi_0=simple_convert_into_pi_from_theta(theta_0)

#epsilon greedy method
def get_action(s,Q, epsilon, pi_0):
    direction=["up", "right", "down", "left"]

    #決定動作
    if np.random.rand()<epsilon:
        #根據epsilon的機率隨機移動
        next_direction=np.random.choice(direction, p=pi_0[s,:])
    else:
        #採用Q為最大值的動作
        next_direction=direction[np.nanargmax(Q[s,:])]

    if next_direction=="up":
        action=0
    elif next_direction=="right":
        action=1
    elif next_direction=="down":
        action=2
    elif next_direction=="left":
        action=3
    return action

def get_s_next(s,a):
    direction=["up", "right", "down", "left"]
    next_direction=direction[a]

    if next_direction=="up":
        s_next=s-3
    elif next_direction=="right":
        s_next=s+1
    elif next_direction=="down":
        s_next=s+3
    elif next_direction=="left":
        s_next=s-1
    return s_next
def Sarsa(s,a,r,s_next, a_next, Q,eta,gamma):
    if s_next==8:
        Q[s,a]=Q[s,a]+eta*(r-Q[s,a])
    else:
        Q[s,a]=Q[s,a]+eta*(r+gamma*Q[s_next,a_next]-Q[s,a])
    return Q

def goal_maze(Q,epsilon,eta,gamma, pi):
    s=0 #起點
    a=a_next=get_action(s,Q,epsilon,pi)
    s_a_history=[[0, np.nan]] #紀錄軌跡

    while 1:
        a=a_next
        s_a_history[-1][1]=a

        s_next=get_s_next(s,a)#取得下一個狀態

        s_a_history.append([s_next,np.nan])

        #給予報酬
        if s_next==8:
            r=1
            a_next=np.nan
        else:
            r=0
            a_next=get_action(s_next,Q,epsilon,pi)

        Q=Sarsa(s,a,r,s_next, a_next,Q,eta,gamma)

        if s_next==8:
            break
        else:
            s=s_next

    return [s_a_history, Q]

#以Sarsa攻克迷宮
eta=0.1
gamma=0.9
epsilon=0.5
v=np.nanmax(Q,axis=1) #計算價值在每個狀態的最大值
is_continue=True
episode=1

while is_continue:
    print("回合："+str(episode))

    #遞減epsilon-greedy的值
    epsilon/=2

    [s_a_history,Q]=goal_maze(Q,epsilon,eta,gamma, pi_0)

    new_v=np.nanmax(Q,axis=1)
    print(np.sum(np.abs(new_v-v)))
    v=new_v

    print("走出迷宮的總步數為"+str(len(s_a_history)-1))

    episode+=1
    if episode>50:
        print(Q)
        break

        
    
