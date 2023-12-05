import numpy as np
import matplotlib.pyplot as plt

# 迷宫的初始位置

# 声明图的大小以及图的变量名
fig=plt.figure(figsize=(5,5))
ax=plt.gca()

# 画出红色的墙壁
# plt.plot([x1,x2],[y1,y2])
plt.plot([1,1],[0,1],color='red',linewidth=2)
plt.plot([1,2],[2,2],color='red',linewidth=2)
plt.plot([2,2],[2,1],color='red',linewidth=2)
plt.plot([2,3],[1,1],color='red',linewidth=2)


# 画出表示状态的的文字
plt.text(0.5,2.5,'S0\nstart',size=14,ha='center')
plt.text(1.5,2.5,'S1',size=14,ha='center')
plt.text(2.5,2.5,'S2',size=14,ha='center')
plt.text(0.5,1.5,'S3',size=14,ha='center')
plt.text(1.5,1.5,'S4',size=14,ha='center')
plt.text(2.5,1.5,'S5',size=14,ha='center')
plt.text(0.5,0.5,'S6',size=14,ha='center')
plt.text(1.5,0.5,'S7',size=14,ha='center')
plt.text(2.5,0.5,'S8\ngoal',size=14,ha='center')


# plt.show()#显示图
# 设定图的范围
ax.set_xlim(0,3)
ax.set_ylim(0,3)
# 设置图的显示
plt.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off',
                right='off',left='off',labelleft='off')

# 当前位置S0用绿色圆圈画出
line,=ax.plot([0.5],[2.5],marker="o",color='g',markersize=60)
# plt.show()

# 策略
# 设定参数θ的初始值theta_0,用于确定初始方案
# S0-S7状态采取的动作,分别为上,右,下,左,不动用np.nan
theta_0=np.array([[np.nan,1,1,np.nan],#S0
                 [np.nan,1,np.nan,1],     #S1
                 [np.nan,np.nan,1,1],#S2
                 [1,1,1,np.nan],     #S3
                 [np.nan,np.nan,1,1],          #S4
                 [1,np.nan,np.nan,np.nan],     #S5
                 [1,np.nan,np.nan,np.nan],#S6
                 [1,1,np.nan,np.nan]      #S7      S8,为目标,无需策略
                 ])



# 动作价值函数以表格形式体现，行表示状态s，列表示动作a，表中的值为动作价值函数Q(s,a)
# 设置初始的动作价值函数
[a,b]=theta_0.shape  #将行列数放入a,b
Q=np.random.rand(a,b)*theta_0 *0.1
#Q为a行b列随机值，乘theta—0是为了使Q的墙壁方向的值为nan
# 如果Q值很大则难以绘图，可通过乘0.1使Q值变小


# 将策略参数theta_0转换为随机策略
def simple_convert_into_pi_from_theta(theta):
    [m,n]=theta.shape#读取theta矩阵的大小
    pi=np.zeros((m,n))
    for i in range(0,m):
        pi[i,:]=theta[i,:]/np.nansum(theta[i,:])#计算比率
    pi=np.nan_to_num(pi)#将nan转换为0
    return pi

#求取随即行动策略pi_0
pi_0=simple_convert_into_pi_from_theta(theta_0)

# 实现贪婪法
# 定义动作函数
def get_actions(s,Q,epsilon,pi_0):
    direction=["up","right","down","left"]
    #确定行动
    if np.random.rand()<epsilon:
        # 以epsilon概率随即行动
        next_direction=np.random.choice(direction,p=pi_0[s,:])
    else:
        # 采用Q的最大值所对应的动作
        next_direction=direction[np.nanargmax(Q[s,:])]

    #为动作加上索引
    if next_direction=="up":
        action = 0
    elif next_direction=="right":
        action=1
    elif next_direction=="down":
        action=2
    elif next_direction=="left":
        action=3
    return action

#将动作作为参数求取下一状态的函数
def get_s_next(s,a,Q,epsilon,pi_0):
    direction=["up","right","down","left"]
    next_direction=direction[a]#采用动作a对应的方向

    #由动作a确定下一个状态
    if next_direction=="up":
        s_next=s-3
    elif next_direction=="right":
        s_next=s+1
    elif next_direction=="down":
        s_next=s+3
    elif next_direction=="left":
        s_next=s-1
    return s_next

#基于Q-learning更新动作价值函数Q
def Q_learning(s,a,r,s_next,Q,eta,gamma):
    #注意到达目标时，下一时刻就不存在
    if s_next==8:#已到达目标
        Q[s,a]=Q[s,a]+eta*(r-Q[s,a])
    else:
        #基于Q-learning更新动作价值函数Q,注意不同，利用下一状态的最大Q值
        Q[s,a]=Q[s,a]+eta*(r+gamma*np.nanmax(Q[s_next,:])-Q[s,a])
    return Q

#定义基于Sarsa求解迷宫问题的函数，输出状态、动作的历史记录以及更新后的Q
def goal_maze_ret_s_a_Q(Q,epsilon,eta,gamma,pi):
    s=0#开始地点
    a=a_next=get_actions(s,Q,epsilon,pi)#初始动作
    s_a_history=[[0,np.nan]]#记录状态、动作历史序列
    while(1):
        a=a_next#更新动作

        s_a_history[-1][1]=a
        #将更新后的动作放在现在的状态下（最终的index=-1）

        s_next=get_s_next(s,a,Q,epsilon,pi)
        #有效的下一状态

        s_a_history.append([s_next,np.nan])
        #把更新的下一状态放入，下一状态的动作未知，设为np.nan

        #给与奖励，求得下一动作
        if s_next==8:
            r=1
            a_next=np.nan
        else:
            r=0
            a_next=get_actions(s_next,Q,epsilon,pi)
            #求下一动作

        #更新价值函数
        Q=Q_learning(s,a,r,s_next,Q,eta,gamma)

        #终止判断
        if s_next==8:
            break
        else:
            s=s_next

    return [s_a_history,Q]


#通过Q_learning求解
eta=0.1#学习率
gamma=0.9#时间折扣率
epsilon=0.5  #epsilon贪婪算法初始值
v=np.nanmax(Q,axis=1)#根据状态求价值的最大
is_continue=True
episode=1

V=[]#用来存放每回合的状态值
V.append(np.nanmax(Q,axis=1))#求各状态下动作价值的最大值

while is_continue:
    print("当前回合为:"+str(episode))

    #epsilon贪婪法的值减少
    epsilon=epsilon/2

    #通过Sarsa求解迷宫问题
    [s_a_history,Q]=goal_maze_ret_s_a_Q(Q,epsilon,eta,gamma,pi_0)

    #状态价值的变化
    new_v=np.nanmax(Q,axis=1)#axis=1表示沿行找最大值
    print(np.sum(np.abs(new_v-v)))
    v=new_v  #添加回合终止时的状态值函数
    V.append(v)
    print("求解迷宫问题所需的步数是："+str(len(s_a_history)-1))

    episode=episode+1
    if episode>100:
        break


# 动画展示
from matplotlib import animation
from IPython.display import HTML
import matplotlib.cm as cm    #color map

def init():
    # 初始化背景图像
    line.set_data([],[])
    return(line,)

def animate(i):
    # 每一帧画面内容
    #各方块中根据状态价值的大小画颜色
    line,=ax.plot([0.5],[2.5],marker="s",color=cm.jet(V[i][0]),markersize=85)#S0
    line,=ax.plot([1.5],[2.5],marker="s",color=cm.jet(V[i][1]),markersize=85)#S1
    line,=ax.plot([2.5],[2.5],marker="s",color=cm.jet(V[i][2]),markersize=85)#S2
    line,=ax.plot([0.5],[1.5],marker="s",color=cm.jet(V[i][3]),markersize=85)#S3
    line,=ax.plot([1.5],[1.5],marker="s",color=cm.jet(V[i][4]),markersize=85)#S4
    line,=ax.plot([2.5],[1.5],marker="s",color=cm.jet(V[i][5]),markersize=85)#S5
    line,=ax.plot([0.5],[0.5],marker="s",color=cm.jet(V[i][6]),markersize=85)#S6
    line,=ax.plot([1.5],[0.5],marker="s",color=cm.jet(V[i][7]),markersize=85)#S7
    line,=ax.plot([2.5],[0.5],marker="s",color=cm.jet(1.0),markersize=85)    #S8
    return line,


# 用初始化函数和绘图函数来绘制动画
anim=animation.FuncAnimation(fig,animate,init_func=init,frames=len(V),interval=200,repeat=False)

HTML(anim.to_jshtml())#在jupyter中显示