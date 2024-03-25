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


#利用softmax函數將策略參數theta轉換成行動策略pi
def softmax_convert_into_pi_from_theta(theta):
    beta=1.0
    [m,n]=theta.shape
    pi=np.zeros((m,n))

    exp_theta=np.exp(beta*theta)
    
    for i in range(m):
        pi[i,:]=exp_theta[i,:]/np.nansum(exp_theta[i,:])

    pi=np.nan_to_num(pi)
    return pi
    
pi_0=softmax_convert_into_pi_from_theta(theta_0)

#epsilon greedy method
def get_action_and_next_s(pi,s):
    direction=["up", "right", "down", "left"]
    #決定動作
    next_direction=np.random.choice(direction, p=pi_0[s,:])

    if next_direction=="up":
        s_next=s-3
        action=0
    elif next_direction=="right":
        s_next=s+1
        action=1
    elif next_direction=="down":
        s_next=s+3
        action=2
    elif next_direction=="left":
        s_next=s-1
        action=3
    return [action,s_next]

def update_theta(theta,pi,s_a_history):
    eta=0.1 #學習率
    T=len(s_a_history)-1

    [m,n]=theta.shape
    delta_theta=theta.copy()

    #於每個元素計算delta_theta
    for i in range(m):
        for j in range(n):
            if not(np.isnan(theta[i,j])):
                #從履歷拿出狀態i的list
                SA_i=[SA for SA in s_a_history if SA[0]==i]

                SA_ij=[SA for SA in s_a_history if SA==[i,j]]

                N_i=len(SA_i)
                N_ij=len(SA_ij)

                delta_theta[i,j]=(N_ij-pi[i,j]*N_i)/T
                
    new_theta=theta+eta*delta_theta
    return new_theta

def goal_maze(pi):
    s=0 #起點
    s_a_history=[[0, np.nan]] #紀錄軌跡

    while 1:
        [action,next_s]=get_action_and_next_s(pi,s)
        s_a_history[-1][1]=action

        s_a_history.append([next_s,np.nan])

        #給予報酬
        if next_s==8:
            break
        else:
            s=next_s
            
    return s_a_history

#以策略梯度法走出迷宮
stop_epsilon=10**(-3)

theta=theta_0
pi=pi_0


is_continue=True
count=1

while is_continue:
    s_a_history=goal_maze(pi)
    new_theta=update_theta(theta,pi,s_a_history)
    new_pi=softmax_convert_into_pi_from_theta(new_theta)

    print(np.sum(np.abs(new_pi-pi)))
    print("總步數為："+str(len(s_a_history)-1))

    if np.sum(np.abs(new_pi-pi))<stop_epsilon:
        is_continue=False
    else:
        theta=new_theta
        pi=new_pi
    
