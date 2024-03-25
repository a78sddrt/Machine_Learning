import random
import numpy as np

#0: scissor, 1:stone, 2:paper

#出剪刀後，下一把要出...出石頭後，下一把要出...出布後，下一把要出
policy_array=np.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]])

playing=15
last_move=np.random.choice([0,1,2])
this_move=np.random.choice([0,1,2])
win_rate=0
def renew_policy(last_move,this_move, win_or_loss):
    if win_or_loss==0:
        policy_array[last_move][(this_move+1)%3]+=0.1*policy_array[last_move][(this_move+1)%3]
        policy_array[last_move,:]/=policy_array[last_move,:].sum()
    elif win_or_loss==1:
        policy_array[last_move][this_move]+=0.1*policy_array[last_move][this_move]
        policy_array[last_move][(this_move+1)%3]-=0.1*policy_array[last_move][(this_move+1)%3]
        policy_array[last_move,:]/=policy_array[last_move,:].sum()
    else:
        policy_array[last_move][this_move]-=0.2*policy_array[last_move][this_move]
        policy_array[last_move][(this_move-1)%3]+=0.2*policy_array[last_move][(this_move-1)%3]
        policy_array[last_move,:]/=policy_array[last_move,:].sum()

        

while(playing>0):
    number_AI=np.random.choice([0,1,2],p=policy_array[this_move])
    last_move=this_move
    this_move=number_AI
    number_player=int(input("Please tell me your choice:"))
    win_or_loss=number_AI-number_player
    if win_or_loss==0:
        renew_policy(last_move,this_move, win_or_loss)
        print("Draw")
        playing-=1
    elif win_or_loss==1 or win_or_loss==-2:
        renew_policy(last_move,this_move, win_or_loss)
        print("You Loss~")
        playing-=1
    else:
        renew_policy(last_move,this_move, win_or_loss)
        print("You Win!")
        win_rate+=1
        playing-=1

print("Winning rate is "+str(win_rate/10))
print(policy_array)



