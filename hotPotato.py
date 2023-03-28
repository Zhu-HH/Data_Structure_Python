"""
约瑟夫问题
"""
from Python_Queue import Queue

def hotPotato(nameList:list,num:int):
    """
    约瑟夫问题,循环报数,数到第N个的玩家退出游戏,直到只剩下一位游戏玩家
    :param nameList: 游戏玩家
    :param num: 循环至第 num 个人的时候 当前玩家退出游戏
    :return: 最后的胜利者
    """
    queue = Queue()
    for name in nameList:
        queue.enqueue(str(name))

    while queue.size()>1:
        for i in range(num-1):
            queue.enqueue(queue.dequeue())
        # print(queue.dequeue())

    return queue.dequeue()

print(hotPotato(["A_name","B_name","C_list","D_dict","E_tuple","F_int","G_str","H_name","I_name","J_anme",],6))