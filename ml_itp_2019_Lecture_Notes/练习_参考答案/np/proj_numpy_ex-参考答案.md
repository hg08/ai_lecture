

```python
# 提示:
#
# 定义函数create_board()：
#  使用numpy 返回一个3x3的array, 初始元素为0，

# 调用该函数， 结果保存到board.并打印出board

#==YOUR CODE===
import numpy as np
def create_board(row=3,col=3):
    return np.zeros((row, col))

create_board()
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
# 提示:
# 
# 1.定义一个函数 place(board, player, position)
# player：代表当前玩家(一个1或2的整数，表示玩家1或2)
# position:一个tuple,表示想要落子的坐标位置
# 只有当position位置未被占用（该位置上的值为0），方能落子，落子后将position位置
# 换成player的值
#
# 2.使用create_board()创建棋盘board,
# 并调用place将玩家1放置到(0,0)


        
# 利用create_board()创建一个棋盘board

board = create_board()

#利用create_board()保存棋盘到变量board, 利用 place()使棋手1放置其自身的号码到位置(0, 0).

def place(board, player, position):
    board[position[0],position[1]]=player
place(board,1,(0,0))

# 打印出board.

print(board)
print('*******')

for row in board:
    print(row)
for element in board.flat:
    print(element,end='| ')
```

    [[1. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    *******
    [1. 0. 0.]
    [0. 0. 0.]
    [0. 0. 0.]
    1.0| 0.0| 0.0| 0.0| 0.0| 0.0| 0.0| 0.0| 0.0| 


```python
# 提示:
# 建立一个函数possibilities(board),返
# 回棋盘上所有未被放置棋子的位置的一个序列(保存为元组).
def possibilities(board):
    returned = []
#     for row_ind,row in enumerate(board):
#         #print(row_ind,row)
    row,col = board.shape
    returned = [(i,j) for i in range(row) for j in range(col) if board[i,j]==0]
    return returned
                
         
possibilities(board)
```




    [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]




```python
# 提示:
# 1.写一个函数random_place(board,player),随机地放置
# 玩家到可能的位置
#
# 2. 可用random.choice()
import random

#==YOUR CODE==

def random_place(board,player):
    rand_pos = random.choice(possibilities(board)) #抽取一个为0的位置
    board[rand_pos[0],rand_pos[1]] = player
    return board
# 调用此函数，并打印board   
#==YOUR CODE==
random_place(board,2)
```




    array([[1., 0., 0.],
           [0., 0., 2.],
           [0., 0., 0.]])




```python
board = create_board()
# 提示：
# 对玩家1和玩家2,调用random_place(board,player)
# 使分别放置3个棋子在棋盘board上.

#==YOUR CODE==
times = 3
for i in range(times):
    random_place(board,1)
    random_place(board,2)
# 打印出board
board
```




    array([[1., 2., 2.],
           [2., 1., 0.],
           [0., 0., 1.]])




```python
# 提示:
# 建立一个函数row_win(board, player)
# 判断如果任意一行满足都是player的标志
# 则返回True,否则返回False
#==YOUR CODE==
def row_win(board, player):
    row,col = board.shape
    temp_array = np.zeros((1,col)) + player # 构造一行数组值为player (也可以直接用单个player的值与数组比较)
    for row_data in board:
        if (temp_array[0] == row_data).all(): # 判断一行数组元素是否完全相等 , temp_array = np.zeros(col) , temp_array == row_data
            return True
#         for c_ind in range(col):
#             if int(row_data[c_ind]) != player:
#                 return False
#         for c_ind in range(col-1):
#             if row_data[c_ind] != row_data[c_ind+1]:
#                 return False
#         if board[row_data[c_ind]] != player:
#             return False
    return False
            
        
# 测试
print(row_win(board,2))
```

    False



```python
# 建立一个函数col_win(board, player)
# 判断如果任意一列满足都是player的标志
# 则返回True,否则返回False

#==YOUR CODE==
def col_win(board, player):
    if row_win(board.T,player):
        return True
    return False
    

# 测试
print(col_win(board,1))
```

    False



```python
# 建立一个函数diag_win(board, player),
# 如果player赢,返回True, 否则False. 
# 可能要用到np.rot90()函数.
# rot90(a)%逆时针旋转90°; rot90(a,-1)%顺时针旋转90°  
"""
0 0 0 0
0 1 0 0
1 1 2 0
0 0 0 0
"""
#==YOUR CODE==
def diag_win_func1(board, player):
    row,col = board.shape
    is_win1 = 0
    is_win2 =0
    
    for r_ind in range(row):
        if int(board[r_ind,r_ind]) == player:
            is_win1 += 1
        if  int(np.rot90(board)[r_ind,r_ind]) == player:
            is_win2 += 1
    if  (is_win1 ==col or is_win2 ==col) and int(board[1,1])!=0:
        return True
    return False

def diag_win(board, player):
    row,col = board.shape
    temp_array = np.zeros(col) + player
    if (board.diagonal() == temp_array).all() or (np.rot90(board).diagonal() == temp_array).all():
        return True
    return False
# 测试
print(board)
print(diag_win(board,1))
```

    [[1. 2. 2.]
     [2. 1. 0.]
     [0. 0. 1.]]
    True



```python
# 提示:
# 创建一个函数evaluate(board)
# 对玩家1,2，判断是否满足`row_win`, `col_win`, 或 `diag_win`
# 若满足，则保存玩家编号到winner
# 若无处落子但还未分出胜负则返回winner=-1

def evaluate(board):
    winner = 0
    for player in [1, 2]:
        # 检查是否 `row_win`, `col_win`, 或 `diag_win` 成立. 
        # 如果是, 保存 `player` 为 `winner`.
        #===YOUR CODE=
        if row_win(board,player) or col_win(board,player) or diag_win(board,player):
            winner = player
        #============= 
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

# 测试
evaluate(board)
```




    1




```python
# 提示:
# 创建一个函数play_game(),实现以下功能:
# 1.创建棋盘
# 2.轮流随机下棋子
# 3.每次落子后使用evaluate()判断输赢
# 4.持续游戏直到出现一个赢家或不可落子
# 5.返回winner:1,2,或-1(表示平局)

# ==YOUR CODE==
row,col=3,3
player1=1
player2=2
def play_game():
    board = create_board(row,col)
    while True:
        random_place(board,player1)
        winner = evaluate(board)
        if winner:
            return winner
        random_place(board,player2)
        winner = evaluate(board)
        if winner:
            return winner
    
# 测试
play_game()

```




    1




```python
import time
import matplotlib.pyplot as plt

# 提示:实现下面的功能
# 1.玩1000次游戏
# 2.计算玩1000次游戏用的时间
# 3.统计1000次游戏的结果存在列表res中
# 4.使用plt.hist(res)观察结果的分布
# 并分析结果：
# 1.玩家1比玩家2赢的次数多吗?
# 2.二玩家赢的次数是否都比平局的次数多?
t1 = time.time()
res = [play_game() for i in range(1000)]
t2 = time.time()
print(t2-t1)
plt.hist(res)
plt.show()

```

    0.6986150741577148



![png](output_10_1.png)



```python
# 提示:
# 编写一个函数play_strategic_game()，实现
# 1.player1先下，player第一次下在棋盘正中心位置,之后轮流随机落子
# 2.返回 winner

# 并调用一次play_strategic_game()
row,col=3,3
player1=1
player2=2
def play_strategic_game():
    board = create_board(row,col)
    place(board,player1,(row//2,col//2))
    while True:
        random_place(board,player2)
        winner = evaluate(board)
        if winner:
            return winner
        random_place(board,player1)
        winner = evaluate(board)
        if winner:
            return winner

play_strategic_game()

```




    1




```python
# 提示:实现下面的功能
# 1.使用play_strategic_game()玩1000次游戏
# 2.计算玩1000次游戏用的时间
# 3.统计1000次游戏的结果存在列表res中
# 4.使用plt.hist(res)观察结果的分布
# 和play_game()做对比，分析结果
# 1.玩家1是否表现得更好了？
# 2.二玩家赢的次数是否都比平局的次数多?
import time
import matplotlib.pyplot as plt

t1 = time.time()
res = [play_strategic_game() for i in range(1000)]
t2 = time.time()
print(t2-t1)
plt.hist(res)
plt.show()

```

    0.5796465873718262



![png](output_12_1.png)



```python
# 100 p
```
