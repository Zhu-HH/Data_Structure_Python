"""
利用栈实现 十进制转二进制
"""
from Python_Stack import Stack


def decimal2bin(number:int):
    stack = Stack()
    while number:
        M = number%2
        stack.push(M)
        number = number// 2
    result_list = []
    while not stack.isEmpty():
        result_list.append(str(stack.pop()))


    result_str =  ''.join(result_list)
    print("原生未处理二进制结果",result_str)

    y = len(result_list)%4
    if y:
        result_str = "0"*(4-y)+result_str
    print("自动 补0 二进制 结果", result_str)
    result_list.reverse()

    # 优化结果 从后往前数4个为一组 不足一组的 自动补充0 中间以空格为分隔符
    result_str_list = []
    for index,strs in enumerate(result_str):
        if (index) %4 !=0 or index ==0:
            result_str_list.append(str(strs))
        else:
            result_str_list.append(str(" "))
            result_str_list.append(str(strs))

    return ''.join(result_str_list)

print("最终效果 ",decimal2bin(178))