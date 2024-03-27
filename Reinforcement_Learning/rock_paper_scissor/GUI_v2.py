import tkinter as tk
from PIL import Image, ImageTk
import random
import numpy as np

#0: scissor, 1:stone, 2:paper

class RockPaperScissorsGame:
    def __init__(self):
   
        #初始化探索指標epsilon
        self.epsilon=0.1
        
        #初始化出拳機率pi
        self.pi=np.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]])
        
        #初始化動作價值函數Q
        self.Q=np.zeros((3,3))
        #初始化遊玩訓練次數episode
        self.episode=0

    
        self.winning_number=0
        self.lossing_number=0
        #初始化獎勵r
        self.r=0
        self.s=0
        self.action=np.random.choice([0,1,2],p=[1/3,1/3,1/3])

    def get_winner(self, user_choice, computer_choice):
        if user_choice == computer_choice:
            return "平手"
        elif (user_choice == '剪刀' and computer_choice == '布') or \
             (user_choice == '石頭' and computer_choice == '剪刀') or \
             (user_choice == '布' and computer_choice == '石頭'):
            return "你贏了！"
        else:
            return "電腦贏了！"
    
    def play_game(self,user_choice,choices):
        self.episode+=1
        r=0
        epsilon=self.epsilon
        pi=self.pi
        Q=self.Q
        episode=self.episode
        s=self.s
        action=self.action

        
        choice=self.get_action(s,Q,pi,epsilon)
        computer_choice = choices[choice]
        s_next=action
        action_next=self.get_action(s_next,Q,pi,epsilon)
    
        result = self.get_winner(user_choice, computer_choice)
        if result=="平手":
            r=0
        elif result=="你贏了！":
            r=-1
            self.winning_number+=1
        else:
            r=1
            self.lossing_number+=1
        #利用sarsa資訊演算法，更新Q(寫成函數sarsa)
        self.Q[s,action]=self.Sarsa(s,s_next,action,action_next,r,Q)
        self.s=s_next
        self.action=action_next
        rate=0
        if self.winning_number+self.lossing_number>0:
            rate=self.winning_number/(self.winning_number+self.lossing_number)
        result_label.config(text=f"你選擇了：{user_choice}\n電腦選擇了：{computer_choice}\n{result}\n\n勝率：{rate:.2f}")

         # 顯示電腦選擇的圖片
        computer_choice_image = Image.open(f"./{computer_choice}.png")
        computer_choice_image = computer_choice_image.resize((50, 50))
        computer_choice_photo = ImageTk.PhotoImage(computer_choice_image)
        computer_choice_label.config(image=computer_choice_photo)
        computer_choice_label.image = computer_choice_photo

    def get_win_rate(self):
        if self.total_games == 0:
            return 0
        return self.user_wins / self.total_games

    def get_action(self,s,Q,pi,epsilon):
        explore=np.random.rand()
        if explore< epsilon:
            return np.random.choice([0,1,2],p=[1/3,1/3,1/3])
        else:
            return np.nanargmax(Q[s,:])
    def Sarsa(self,s, s_next, a, a_next,r,Q):
        gamma=0.99
        new=Q[s,a]+0.4*(r+gamma*Q[s_next,a_next]-Q[s,a])
        return new

# 創建遊戲實例
game = RockPaperScissorsGame()

# 創建主視窗
root = tk.Tk()
root.title("剪刀石頭布遊戲")

# 選擇框架
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# 選擇標籤
choice_label = tk.Label(frame, text="請選擇：")
choice_label.pack()

# 創建圖片按鈕
choices = ['剪刀', '石頭', '布']
images = []
for i, choice in enumerate(choices):
    image = Image.open(f"{choice}.png")  # 圖片文件名為 剪刀.png, 石頭.png, 布.png
    image = image.resize((50, 50))  # 調整圖片大小
    photo = ImageTk.PhotoImage(image)
    images.append(photo)
    button = tk.Button(frame, image=photo, command=lambda c=choice: game.play_game(c,choices))
    button.image = photo  # 保留對圖片對象的引用，避免被垃圾回收
    button.pack(side=tk.LEFT, padx=5, pady=5)
    # 縮放圖片
    button.config(width=50, height=50)

# 電腦選擇圖片
computer_choice_label = tk.Label(root, text="電腦選擇：")
computer_choice_label.pack()
computer_choice_image_label = tk.Label(root)
computer_choice_image_label.pack()

# 結果標籤
result_label = tk.Label(root, text="")
result_label.pack(padx=20, pady=10)

# 運行主迴圈
root.mainloop()
