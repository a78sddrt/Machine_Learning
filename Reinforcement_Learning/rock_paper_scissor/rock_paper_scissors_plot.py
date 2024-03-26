import random
import numpy as np
import matplotlib.pyplot as plt

#0: scissor, 1:stone, 2:paper

eta=0.2

def get_action(s,Q,pi,epsilon):
    explore=np.random.rand()
    if explore< epsilon:
        return np.random.choice([0,1,2],p=[1/3,1/3,1/3])
    else:
        return np.nanargmax(Q[s,:])
def Sarsa(s, s_next, a, a_next,r,Q):
    gamma=0.9
    Q[s,a]=Q[s,a]+eta*(r+gamma*Q[s_next,a_next]-Q[s,a])
    return Q


def Learning(epsilon):
    episode=0
    rate=[]
    #初始化出拳機率pi
    pi=np.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]])
    #初始化動作價值函數Q
    Q=np.zeros((3,3))

    winning_number=0
    lossing_number=0
    #初始化獎勵r
    r=0
    s=0
    action=np.random.choice([0,1,2],p=pi[s])

    #開始玩遊戲：
    while episode<2000:
    #策略函數根據現在狀態，Q和pi，得到下一個動作(寫成函數get_action)
        action=get_action(s,Q,pi,epsilon)
        epsilon=epsilon/(episode+1)
    #利用現在的狀態和動作，得到下一個狀態&動作(這裡是下一個狀態與動作相同）
        s_next=action
        action_next=get_action(s_next,Q,pi,epsilon)
    #與玩家遊玩：獲得獎勵(贏＋1, 輸-1)
        number_player=np.random.choice([0,1,2],p=[1/4,1/4,1/2])
        win_or_loss=action-number_player
        if win_or_loss==1 or win_or_loss==-2:
            r=1
            winning_number+=1
        elif win_or_loss==0:
            r=-1
            continue
        else:
            r=-1
            lossing_number+=1
    #利用sarsa資訊演算法，更新Q(寫成函數sarsa)
        Q=Sarsa(s,s_next,action,action_next,r,Q)
        s=s_next
        #印出目前勝率
        if winning_number+lossing_number>0:
            print("AI出：{}".format(action))
            print(winning_number/(winning_number+lossing_number))
            rate.append(winning_number/(winning_number+lossing_number))
    #重複，直到次數達到遊玩次數episode
        episode+=1
    return rate

for i in range(0,5):
    rate=Learning((i+1)*0.1)
    plt.plot(rate,label=(i+1)*0.1)
plt.legend()
plt.show()
















