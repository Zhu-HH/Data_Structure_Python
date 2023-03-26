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
str_left = ["(","{","["]
str_right = [")","}","]"]
def parenthesisMatching(str_data):
    for index,strs in enumerate(str_data):
        if strs in str_left:
            stack.push(strs)
        else:
            if strs in str_right and not stack.isEmpty():
                pop_data = stack.pop()
                if str_dic[pop_data] != strs:
                    return False
                else:
                    continue
            else:
                if stack.isEmpty() and index == len(str_data)-1:
                    return False
                else:
                    continue
    else:
        if stack.isEmpty():
            return True
        else:
            return False
print(parenthesisMatching('{([()()()((((}()))))])}'))
# print(parenthesisMatching('(1+1)-(){{}{}}'))
# print(parenthesisMatching('(1+1)-(1+1)*{{1+1}-{1+1}}'))
# print(parenthesisMatching(']'))
# print(parenthesisMatching('[()0}(){}]'))

