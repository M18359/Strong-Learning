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
# # 将参数θ转换为策略Π,即转换为概率
# def simple_convert_into_pi_from_theta(theta):#     简单计算百分比
#     [m,n]=theta.shape
#     pi=np.zeros((m,n))
#     for i in range(0,m):
#         pi[i,:]=theta[i,:]/np.nansum(theta[i,:])#计算百分比
#     pi = np.nan_to_num(pi)               #把pi中nan转换为0
#     return pi

# 参数theta利用softmax函数计算概率转换为行动策略Π的定义
def softmax_convert_into_pi_from_theta(theta):
    beta=1.0
    [m,n]=theta.shape
    pi=np.zeros((m,n))
    exp_theta=np.exp(beta*theta)
    for i in range(0,m):
        pi[i,:]=exp_theta[i,:]/np.nansum(exp_theta[i,:])
    pi=np.nan_to_num(pi)
    return pi

# 采用softmax ,定义求取动作a以及1步移动后到达状态
def get_action_and_next_state(pi,s):
    direction=["up","right","down","left"]
    next_direction=np.random.choice(direction,p=pi[s,:])
    if next_direction=="up":
        action=0
        s_next = s-3
    elif next_direction=="right":
        action=1
        s_next = s+1
    elif next_direction=="down":
        s_next=s+3
        action=2
    elif next_direction=="left":
        s_next=s-1
        action=3
    return [action,s_next]

# 定义theta更新函数,利用策略梯度法
def up_date_theta(theta,pi,s_a_history):
    eta=0.1#定义学习率
    T=len(s_a_history)-1  #到达目标的总步数
    [m,n]=theta.shape
    delta_theta=theta.copy() #生成初始的delta_theta,由于指针原因不能直接使用delta_theta=theta

#     求取delta_theta的各元素
    for i in range(0,m):
        for j in range(0,n):
            if not(np.isnan(theta[i,j])):#theta不是nan时:
                SA_i=[SA for SA in s_a_history if SA[0]==i]#从状态列表中取出状态i
                SA_ij=[SA for SA in s_a_history if SA==[i,j]]#取出状态i下应采取的动作j
                N_i=len(SA_i)#状态下动作的总次数
                N_ij=len(SA_ij)#状态i下采取动作j的次数
                delta_theta[i,j]=(N_ij-pi[i,j]*N_i)/T     #更新规则
    new_theta=theta+eta*delta_theta
    return new_theta





# 定义迷宫内使机器人持续移动的函数,输出历史状态和动作
def goal_maze_ret_s_a(pi):
    s=0;
    s_a_history=[[0,np.nan]]#记录智能体采取动作和移动轨迹的列表
    while(1):
        # 运动一步至下一状态
        [action,next_s]=get_action_and_next_state(pi,s)
        s_a_history[-1][1]=action    #带入当前状态(即目标最后一个状态index=-1)的动作
                                     # a[-1]表示a队列中最后一位
        # 记录这一状态,由于还不知道其动作,用nan表示
        s_a_history.append([next_s,np.nan])
        # 判断是否已到达终止状态，是则跳出，否则更新当前状态，再进行下一步运动
        if next_s==8:
           break
        else:
            s=next_s
    return s_a_history



# pi_0=softmax_convert_into_pi_from_theta(theta_0)
# # print(pi_0)
# s_a_history=goal_maze_ret_s_a(pi_0)
# # print(s_a_history)
# # print("求解迷宫路径所需的步数是"+str(len(s_a_history)-1))
# new_theta=up_date_theta(theta_0,pi_0,s_a_history)
# pi=softmax_convert_into_pi_from_theta(new_theta)
# print(pi)

# 主程序实现
stop_epsilon = 10**-4  #变化策略小于10^-4则结束学习

theta=theta_0
pi_0=softmax_convert_into_pi_from_theta(theta_0)
pi=pi_0

is_continue = True
count=1
while is_continue:
    s_a_history=goal_maze_ret_s_a(pi)#由策略pi搜索迷宫探索历史
    new_theta=up_date_theta(theta,pi,s_a_history)
    new_pi=softmax_convert_into_pi_from_theta(new_theta)

    # print(np.sum(np.abs(new_pi-pi)))#输出策略的变化
    # print("求解迷宫问题所需要的步数:"+str(len(s_a_history)-1))

    if np.sum(np.abs(new_pi-pi))<stop_epsilon:
        is_continue=False
    else:
        theta=new_theta
        pi=new_pi
# 确认最终策略
np.set_printoptions(precision=3,suppress=True)
print(pi)


# 动画展示
from matplotlib import animation
from IPython.display import HTML

def init():
    # 初始化背景图像
    line.set_data([],[])
    return(line,)

def animate(i):
    # 每一帧画面内容
    state=s_a_history[i][0]   #画出当前位置,i为1-8任意位置
    # 设置绿色圆圈中心坐标
    # x与状态间关系
    x=(state % 3)+0.5
    # y与状态间关系
    y=2.5-int(state/3)
    line.set_data(x,y)
    return (line,)

# 用初始化函数和绘图函数来绘制动画
anim=animation.FuncAnimation(fig,animate,init_func=init,frames=len(s_a_history),interval=200,repeat=False)

HTML(anim.to_jshtml())#在jupyter中显示