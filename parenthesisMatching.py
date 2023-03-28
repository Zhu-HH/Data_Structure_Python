"""
利用栈实现括号匹配
"""
from Python_Stack import Stack
stack = Stack()

str_dic = {
    "(":")",
    "[":"]",
    "{":"}",
}
# 左括号与右括号
str_left = ["(","{","["]
str_right = [")","}","]"]

def parenthesisMatching(str_data):
    """
    输入字符串,判断括号是否匹配
    :param str_data: {([()()()((((}()))))])}
    :return: True
    """
    for index,strs in enumerate(str_data):
        if strs in str_left:
            stack.push(strs)
        else:
            if strs in str_right and not stack.isEmpty():   # 此判断 有效解决了只出现一个右括号的问题
                pop_data = stack.pop()
                if str_dic[pop_data] != strs:
                    return False
                else:
                    continue
            else:
                if stack.isEmpty() and (strs not in str_right) and index == len(str_data)-1:
                    return True
                elif strs in str_right:
                    return False
    else:
        if stack.isEmpty():
            return True
        else:
            return False

# print(parenthesisMatching(']'))
# print(parenthesisMatching('['))
# print(parenthesisMatching('[(])'))
print(parenthesisMatching('{([()()()((((()))))])}'))
# print(parenthesisMatching('[(1)(2){3*3}]'))
print(parenthesisMatching('(1+1)-(){{}{}}'))
print(parenthesisMatching('(1+1)-(1+1)*{{1+1}-{1+1}}+1'))



