"""
Python实现栈的功能
"""
class Stack():
    """
    自定义实现栈
    """
    def __init__(self,list_item=None):
        self.items = list_item if list_item else []

    @classmethod
    def from_list(cls,list_item:list):
        return cls(list_item=list_item)

    def isEmpty(self):
        return self.items == []

    def push(self,item):
        self.items.append(item)

    def pop(self):
        return self.items.pop() if self.items else None

    def peek(self):
        return self.items[-1] if self.items else None

    def size(self):
        return len(self.items)

if __name__ == '__main__':
    stack_1 = Stack()
    stack_2 = Stack([1,2,3,4,'a','b'])
    print(stack_1.pop(),stack_2.size())
