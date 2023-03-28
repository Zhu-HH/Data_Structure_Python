"""
中缀表达式转后缀表达式
基本流程:
    1. 当前token 是操作数(英文字母或数字) 直接加入到后缀表达式的列表中
    2. 若不是操作数,而是 左括号( 意味着接下来直到遇到 右括号) 才能结束这一个小单元 需要压入栈中来做判断
    3. 当遇到了 右括号) 就需要在栈中弹出 操作运算符然后加入到后缀表达式的列表中, 直到遇到左括号停止
    4. 当前扫描到了 操作运算符 +-*/  需要判断
        a. 若当前扫描运算符的优先级 小于或等于栈顶运算符的优先级 需要将栈顶运算符弹出来加入后缀表达式列表中 直到 栈空 或 栈顶的优先级比
            当前扫描的运算符优先级低 停止扫描
        b. 如果不满足a 直接将当前运算符推入栈中
    5. 当扫描完毕时, 若栈内还有操作符 就依次弹出来 写入后缀表达式的列表中
"""
from Python_Stack import Stack


def infix2Postfix(infix_str: str):
    # 定义运算符优先级
    prec = {
        "*": 3,
        "/": 3,
        "+": 2,
        "-": 2,
        "(": 1,
    }
    opStack = Stack()
    postfixList = []
    tokenList = infix_str.split()
    for token in tokenList:
        if token.isdigit() or token.isupper():
            postfixList.append(token)
        elif token == '(':
            opStack.push(token)
        elif token == ')':
            toptoken = opStack.pop()
            while toptoken != '(':
                postfixList.append(toptoken)
                toptoken = opStack.pop()
        else:
            while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
                postfixList.append(opStack.pop())
            opStack.push(token)
    while not opStack.isEmpty():
        postfixList.append(opStack.pop())

    return ' '.join(postfixList)

print(infix2Postfix('( A + ( B * C ) ) * D'))
