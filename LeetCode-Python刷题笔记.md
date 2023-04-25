### 第一题  两数之和 简单
#### 算法描述

1. 利用Python Dict的特性, nums数组中的元素作为key,数组nums下标索引作为value. 创建一个字典用于存放元素与索引下标对应关系.
2. 同时借助Python内置方法enumerate(nums),该内置方法会返回下标索引index,nums[index] 数据.
3. 依次遍历nums中每个元素,判断 target - nums[index] 是否存在Dict中. 判断当前处理元素与target的差值是否存在Dict中?
    + target - nums[index] 存在Dict中,直接返回Dict中 target - nums[index] 对应的value值以及当前index值,程序结束.
    + target - nums[index] 不在Dict中,将当前nums[index] 作为key,index 作为 value, 存入Dict中.继续遍历nums.

#### 算法实现

##### Python算法实现

```python
def twoSum(nums: list, target: int) -> list:
    data_dict = dict()
    for index, num in enumerate(nums):
        if data_dict.get(target - num,None) is not None:
            return [data_dict[target - num], index]
        else:
            data_dict[num] = index
```
##### C++算法实现

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> head;                  // 对应Python中的 Dict
        for(int i=0;i<nums.size();i++){
            int t = target - nums[i];
            if(head.count(t)){
                return {head[t],i};
            }else{
                head[nums[i]] = i;
            }
        }
        return {};
    }
};
```


### 第二题  两数相加 中等

#### 算法描述

1. 链表结构, 输入两个链表 2 -> 4 -> 3,5 -> 6 -> 4  两个链表相加 7-> 0 -> 8 
2. 依次遍历两个链表,依次相加对应链表的值以及进位,直到其中一条链表为空结束遍历. 
3. 判断链表不为空,从第二步断的位置接着开始遍历并累加.

#### 算法实现

##### Python算法实现

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode(None)
        previous = head
        r = 0
        while l1 is not None  and l2 is not None:
            temp = ListNode(None)
            previous.next = temp
            previous = temp
            val = l1.val + l2.val + r
            previous.val = val % 10
            r = val // 10
            l1 = l1.next
            l2 = l2.next
        
        while l1:
            temp = ListNode(None)
            previous.next = temp
            previous = temp
            val = l1.val + r
            previous.val = val % 10
            r = val // 10
            l1 = l1.next
        
        while l2:
            temp = ListNode(None)
            previous.next = temp
            previous = temp
            val = l2.val + r
            previous.val = val % 10
            r = val // 10
            l2 = l2.next

        if r > 0:
            temp = ListNode(r)
            previous.next = temp
            previous = temp
        
        return head.next
```
##### C++ 算法实现

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        auto head = new ListNode(-1),cur = head;     // 新创建一个头节点,使链表成为有头节点的链表,这样操作表头结点不需要额外判断
        int t = 0;
        while(l1 || l2 || t){
            if(l1) t+=l1->val,l1=l1->next;
            if(l2) t+=l2->val,l2=l2->next;
            cur = cur->next = new ListNode(t%10);
            t /=10;
        }
        return head->next;
    }
};
```

### 第三题  无重复字符的最长子串 中等

#### 算法描述

1. 将字符串转为列表,方便操作.
2. 采用双指针形式进行操作,previous代表检查指针,index代表当前遍历指针.
3. 依次遍历,index 指向的元素不存在dict中,就加入并且数值设为1.
4. index指向的元素存在,对应value就+=1,之后判断previous所指向的元素在dict中的值是否大于1 且 previous 小于 index
    + 若满足条件, 就说明目前为止 已经达到最大的字串
    + 需要previous 依次向后检查是否是超出数量的字符
        + dict中对应字符减去1 且 previous +=1
        + 既然是减去 则不可能去执行 max 操作 
5. 直到index运行结束 返回 

#### 算法实现

##### Python算法实现1: List 实现 代码易读 但效率不高
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_length_str = []   # 用来保存当前最长字串
        max_length = 0
        for str in list(s):
            if str in max_length_str:   # 说明当前 str 不满足条件了 需要检查
                # 删除出现重复字符前所有的字符 包含冲突字符 
                while str in max_length_str:
                    max_length_str.pop(0)
                max_length_str.append(str)
                if max_length < len(max_length_str) :max_length = len(max_length_str) 
            else:
                max_length_str.append(str)
                if max_length < len(max_length_str) :max_length = len(max_length_str)
        return max_length
```

##### Python算法实现2 : 双指针解决 效率中等偏上

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        _dict = {}
        _length = 0
        previous  = 0               # 发生冲突,用来检查当前字符是否为冲突字符
        for index in range(len(s)):
            if not _dict.get(s[index],None):
                _dict[s[index]] = 1
                _length = max(_length, index - previous + 1)
            else:
                _dict[s[index]] += 1
                while _dict.get(s[index]) > 1:
                    _dict[s[previous]] -= 1
                    previous += 1
        return _length
```
##### C++ 算法实现 
```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int>heap;
        int res = 0;
        for(int i = 0,j = 0;j < s.size();j++){
            heap[s[j]]++;
            while(heap[s[j]]>1) heap[s[i++]]--;       // 这里的 i 指针 代表着 Python代码中的previous指针
            res = max(res,j-i+1);
        }
        return res;
    }
};
```

### 第四题  寻找两个正序数组的中位数 困难


#### 算法描述

1. Python 实现这道题简单,只需几个操作就可以返回值
2. 


#### 算法实现

##### Python 算法实现
```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums1 += nums2
        nums1.sort()
        half_len = len(nums1)//2
        if len(nums1) %2:
            return nums1[half_len]
        return (nums1[half_len] + nums1[half_len - 1]) /2
```

##### C++ 算法实现
```c++
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int balance = (nums1.size()+nums2.size()) >> 1;  // 获取 数组中间索引
        int flag = (nums1.size()+nums2.size())%2;  // 是否为奇数 奇数则返回 索引 和索引+1 的值
        int count = 0;
        vector<int>::iterator it1 = nums1.begin();
        vector<int>::iterator it2 = nums2.begin();
        vector<double> new_nums;
        while(it1!=nums1.end() || it2!=nums2.end()){
            if(it1!=nums1.end() && it2!=nums2.end()){
                if(*it1 > *it2){
                    new_nums.push_back(*it2);
                    it2++;
                    continue;
                }else{
                    new_nums.push_back(*it1);
                    it1++;
                    continue;
                }
            }
            if(new_nums.size()>=balance+1) break;
            if(it1!=nums1.end()){new_nums.push_back(*it1);it1++;}
            if(it2!=nums2.end()){new_nums.push_back(*it2);it2++;}
        }
        return flag?new_nums[balance]:(new_nums[balance-1]+new_nums[balance])/2;
    }
};
```
### 第五题  最长回文子串   中等
#### 算法描述

1. 有两种类型的最长回文子串
    + 第一种 总个数为奇数的对称形式回文子串    abcba
    + 第二种 总个数为偶数的对称形式回文子串    abba
2. 采用双指针来检查左端和右端是否一致
3. 


#### 算法实现

##### Python 算法实现

```python
 class Solution:
    def check(self,l,r,res,s,_len):
        while(l >= 0 and r < _len and s[l] == s[r]):
                l -= 1
                r += 1
        if (len(res) < r - l - 1):
            res = s[l+1:r]
        return res
    def longestPalindrome(self, s: str) -> str:
        res = ""
        _len = len(s)
        for i in range(_len):
            res = self.check(i-1,i+1,res,s,_len)
            res = self.check(i,i+1,res,s,_len)
        return res
```

##### Pythonic 实现

```python

```



##### C++ 算法实现
```c++
class Solution {
public:
    string longestPalindrome(string s) {
        string res;
        for (int i=0;i<s.size();i++){
            int l = i-1,r = i+1;// 匹配是否为 奇数的回文子串
            while(l>=0 && r<s.size() && s[l] == s[r]) l--,r++;   
            if(res.size() < r-l-1) res = s.substr(l+1,r-l-1);   // 此处  r-l-1 代表的是 退出while循环之前的下标  也就是 回文子串的个数 
            l = i,r=i+1; // 匹配是否为 偶数的回文子串
            while(l>=0 && r<s.size() && s[l] == s[r]) l--,r++;  
            if(res.size() < r-l-1) res = s.substr(l+1,r-l-1);
        }
        return res;
    }
};
```

### 第六题  N字形变换 中等
#### 算法描述
纯数学公式解决此题  叫做  公差为(numRows - 1)*2 等差数列...
1. 首先计算出来一个固定的差值,类似于斐波那契数列,只不过此题 需要每一行前两个的数据 以及一个固定的差值就可以得出每一行字符串的下标索引
2. 固定差值 = ( numRows - 1 ) * 2
3. 程序思路是 先计算出每一行的下表索引,然后最后根据索引输出最后的字符串
4. 以 numRows = 5 为例子分析
    + 最终输出第 1 行字符串: s[0] + s[0 + (numRows - 1 ) * 2],此时 第二个下标等于 第一个下标 + 固定算法 , 第三个下标等于第一个下标+固定值...  static = 8   **这叫 差为8的等差数列**
        + s[0] + s[8] + s[16] + s[24]   第一行特殊 不需要提前知道两个才可以计算
        + [0,8,16,24,...]
    + 最终输出第 2 行字符串: s[1] + s[1 + (numRows - 2 ) * 2]
        + s[1] + s[1 + 6] + s[9] + s[15] ...
        + [1,7,9,15,...] 
    + 最终输出第 3 行字符串: s[2] + s[2 + (numRows - 3 ) * 2]
        + s[2] + s[6] + s[10] + s[14] ...
        + [2,6,10,14,...]
    + 最终输出第 4 行字符串: s[3] + s[3 + (numRows - 4 ) * 2]
        + s[3] + s[5] + s[11] + s[13] ...
        + [3,5,11,13,...]
    + 最终输出第 5 行字符串: s[4] + s[4 + (numRows - 5 ) * 2]
        + s[4] + s[12] + s[20] ... 
        + [4,12,20,...]
5. 不难发现,除开最后一行外,每行都是前两个索引相加为 公差  i = 行数, k = (_static - i)  
6. 吧全部索引相加到一个list中 然后最后 还原字符串结束 

#### 算法实现

##### Python 算法实现  效率太低了 
```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1: return s
        _len = len(s)
        if len(s) <= numRows: return s
        _static = (numRows - 1) * 2
        _list = list(filter(lambda x: x % _static == 0, [i for i in range(_len)]))
        for index in range(1, numRows - 1):
            _list.append(index)
            _list.append(index * 3 if index == numRows - 1 else _static - index)
            i = -2
            while _list[i] + _static < _len:
                _list.append(_list[i] + _static)
        for i in range(1, _len, 2):
            if (numRows - 1) * i < _len:
                _list.append((numRows - 1) * i)
        _str = ""
        for index in _list:
            if index < _len:
                _str += s[index]
        return _str
```
##### C++ 风格的Python代码实现 效率不错

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or len(s) <= numRows: return s
        _len = len(s)
        _static = (numRows - 1) * 2
        _str = ""
        for start in range(numRows):
            if start == 0 or start == numRows -1:
                for index in range(start,_len,_static):
                    _str += s[index]
            else:
                i = start
                j = _static -  i
                while (i < _len or j < _len): 
                    if i < _len: 
                        _str += s[i]
                        i += _static
                    if j < _len: 
                        _str += s[j]
                        j += _static
        return _str
```

##### C++ 算法实现 
```c++
class Solution {
public:
    string convert(string s, int numRows) {
        if (numRows == 1 || numRows >= s.size()) return s;
        string res;
        int _static = ( numRows - 1 ) * 2;
        for(int i = 0;i < numRows;i ++ ){
            if(i == 0 || i == numRows - 1){
                for (int j = i;j < s.size() ; j += _static ) res += s[j];

            }else{
                for (int j = i, k = _static - i;j < s.size() || k < s.size(); j += _static,k += _static){
                    if (j < s.size()) res += s[j];
                    if (k < s.size()) res += s[k];
                }

            }
        }
        return res;
    }
};
```

### 第七题  整数反转 

#### 算法描述 

1. 先将整数取绝对值 再转为字符串
2. 从后往前开始判断
    + 若 当前不为0  或者 _存在   就将当前字符作为有效字符
3. 最后判断是否超过界限

#### 算法实现
##### Python 算法实现

```python
class Solution:
    def reverse(self, x: int) -> int:
        if x == 0: return 0
        flag = -1 if x < 0 else 1
        _ = 0
        str_x = str(abs(x))
        for i in range(len(str_x)-1,-1,-1):
            if str_x[i] or _:           # 有效解决了 90010 倒过来的情况
                _ = (int(str_x[i]) + _ * 10)
        if abs(_ * flag) > 2**31 - 1:
            return 0
        return _ * flag
```



##### C++ 算法实现


### 第八题  字符串转整数 中等

#### 算法描述

#### 算法实现
##### Python算法实现 使用re
```python
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1), -2**31)
```
#### Python 算法实现 遍历   奇奇怪怪测试数据集 暂时跳过
```python

```

### 第九题  回文数 

#### 算法描述 

转为字符串 处理简单 直接看代码

#### 算法实现 

##### Python 算法实现 效率高

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x<0:return False
        return str(x) == str(x)[::-1
```

### 第十题 正则表达式匹配    不会 跳过

#### 算法描述



#### 算法实现

##### Python 算法实现



### 第二十题 有效的括号  简单 

#### 算法描述 
1. 利用栈的特性来实现括号匹配,先对s进行list转换
2. 依次遍历,若为左括号 入栈 若为右括号,出栈并且检查括号是否为一组
3. 处理只有一个右括号的形式 以及处理 为右括号且此时栈为空的情况

#### 算法实现

##### Python算法实现  效率中等
```python
class Solution:
    def isValid(self, s: str) -> bool:
        left_data = "([{"
        right_data = ")]}"
        stack = []
        for str in list(s):
            if str in left_data:
                stack.append(str)
            elif stack != []:
                if left_data.index(stack.pop()) != right_data.index(str):
                    return False
            else:
                return False
        return stack == []
```



### 第六七四题 最长连续递增序列

#### 算法描述

dp思想来解题就会很快

#### 算法实现

##### Python 算法实现

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        _ = [-10**10]
        _max = 0
        for data in nums:
            if data > _[-1]:
                _.append(data)
                _max = max(_max,len(_)if _[0]!= -10**10 else len(_)-1)
            elif data <= _[-1]:
                _ = [data]
        return _max
```



### 第二四一八题 按身高排序

#### 算法描述

根据题目的 互不相同的身高作为 dict的key 进行排序

#### 算法实现

##### Python 算法实现1 效率低
```python
class Solution:
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        _ = heights[:]
        heights.sort(reverse=True)
        _list = []
        for i in heights:
            _list.append(names[_.index(i)])
        return _list
```

##### Python 算法实现2  效率中等 

```python
class Solution:
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        _dict = dict(zip(heights,names))
        _dict = dict(sorted(_dict.items(),key = lambda x:x[0],reverse = True))
        return list(_dict.values())
```

##### Pythonic 风格代码
```python
class Solution:
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        return [name for height, name in sorted(zip(heights, names), reverse=True)]
```

