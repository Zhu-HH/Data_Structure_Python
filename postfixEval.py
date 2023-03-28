"""
后缀表达式求值
后缀表达式的特性 必是 两个擦作数 一个运算符 这样的结构
基本流程:
    1. 将表达式依次读取入栈,若当前扫描的为运损操作符 就在栈内弹出两个数据进行运算 运算完成后 将结果 推入栈中继续向后操作

"""

from Python_Stack import Stack
def postfixEval(postFix :str):
    stack = Stack()
    op = ["+","-","*","/"]
    for data in postFix.replace(" ",''):
        if data in op:
            M = stack.pop()
            N = stack.pop()
            result = eval("{}{}{}".format(N,data,M))
            stack.push(result)
        else:
            stack.push(data)
    return stack.peek()

print(postfixEval('1 2 3 * + 7 *'))