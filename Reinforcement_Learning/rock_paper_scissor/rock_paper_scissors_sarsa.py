import random
import numpy as np

#0: scissor, 1:stone, 2:paper



def get_action(s,Q,pi,epsilon):
    explore=np.random.rand()
    if explore< epsilon:
        return np.random.choice([0,1,2],p=pi[s])
    else:
        return np.nanargmax(Q[s,:])
def Sarsa(s, s_next, a, a_next,r):
    gamma=0.9
    Q[s,a]=Q[s,a]+0.1*(r+gamma*Q[s_next,a_next]-Q[s,a])
    return Q


#初始化探索指標epsilon
epsilon=0.3
#初始化出拳機率pi
pi=np.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]])
#初始化動作價值函數Q
Q=np.zeros((3,3))
#初始化遊玩訓練次數episode
episode=1

winning_number=0
lossing_number=0
#初始化獎勵r
r=0
s=0
action=np.random.choice([0,1,2],p=pi[s])

#開始玩遊戲：
while 1:
#策略函數根據現在狀態，Q和pi，得到下一個動作(寫成函數get_action)
    action=get_action(s,Q,pi,epsilon)
    
#利用現在的狀態和動作，得到下一個狀態&動作(這裡是下一個狀態與動作相同）
    s_next=action
    action_next=get_action(s_next,Q,pi,epsilon)
#與玩家遊玩：獲得獎勵(贏＋1, 輸-1)
    number_player=int(input("Please tell me your choice:"))
    win_or_loss=action-number_player
    if win_or_loss==0:
        print("Draw")
    elif win_or_loss==1 or win_or_loss==-2:
        r=-1
        lossing_number+=1
        print("You Loss~")
    else:
        r=1
        winning_number+=1
        print("You Win!")

#利用sarsa資訊演算法，更新Q(寫成函數sarsa)
    Q=Sarsa(s,s_next,action,action_next,r)
    s=s_next
#印出目前勝率
    if winning_number+lossing_number>0:
        rate=winning_number/(winning_number+lossing_number)
        print(rate)
#重複，直到次數達到遊玩次數episode
    episode+=1



















