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

> 同向双指针  滑动窗口   题三

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
##### Python算法实现3 : 同向指针 

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        _res = 0
        cnt = Counter()  # hashmap key-> char value -> int
        left = 0
        for right,c in enumerate(s):
            cnt[c] += 1
            while (cnt[c] > 1):
                cnt[s[left]] -= 1
                left += 1
            _res = max(_res,right - left + 1)
        return _res
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



### 第十一题  盛最多水的容器 

> 接雨水问题 题一

#### 算法描述

1. 采用双指针形式,左指针i 右指针j.
2. 结果 等于 下标 j-i 乘 对应指针值的最小值
3. 当左指针的值小于右指针的时候 左指针 向后移动  否则 右指针向前移动
4. 当 左指针 大于右指针的时候就 说明结束了循环

#### 算法实现

##### C++ 算法实现

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int res = 0;
        for (int i = 0,j = height.size() - 1;i < j;){
            res = max(res,(j-i)*min(height[i],height[j]));
            if ( height[i] < height[j]) i++;else j--;
        }
        return res;
    }
};
```

##### Python 算法实现

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        i,j,res = 0,len(height)-1,0
        while i<j:
            res = max(res,min(height[i],height[j])*(j-i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return res
```

### 第十二题 整数转罗马数字

#### 算法描述

1. 找规律,提前建立好一一对应关系,然后方便去映射
    + `_dict = {1000:'M',900:'CM',500:'D',400:'CD',100:'C',90:'XC',50:'L',40:'XL',10:'X',9:'IX',5:'V',4:'IV',1:'I'}`
2.  让 num 依次对 keys[i] 取余,若有结果K,说明需要K个对应的value,最后 对 num 减去 K*keys[i] 
3. 一直重复2  直到 num =0 结束 

#### 算法实现

##### C++ 算法实现

```c++
class Solution {
public:
    string intToRoman(int num) {
        string res = "";
        int keys[] = {1000,900,500,400,100,90,50,40,10,9,5,4,1};
        string values[] = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        for(int i=0;i<13;i++){
            while(num >= keys[i]){
                num -= keys[i];
                res += values[i];
            }
        }
        return res;
    }
};
```



##### Python算法实现

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        _dict = {1000:'M',900:'CM',500:'D',400:'CD',100:'C',90:'XC',50:'L',40:'XL',10:'X',9:'IX',5:'V',4:'IV',1:'I'}
        res = ""
        while num:
            for _ in _dict.keys():
                __ = num // _
                if __:
                    res += (_dict[_] * __)
                    num -= (_ * __)
        return res
```

 ### 第十三题 罗马数字转整数

#### 算法描述

是上一题的逆运用 整体思路差不多

需要对字符串s 判断一下  当前字符串 和 下一个字符串组成的新字符串是否存在映射表中

#### 算法实现

##### Python 算法实现 

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        _dict,i,res,_len = {"M":1000,"CM":900,"D":500,"CD":400,"C":100,"XC":90,"L":50,"XL":40,"X":10,"IX":9,"V":5,"IV":4,"I":1},0,0,len(s)
        while i < _len:
            if _dict.get(s[i]):
                if _dict.get(s[i:i+2]):
                    res += (_dict[s[i:i+2]])
                    i += 2
                else:
                    res += _dict[s[i]]
                    i += 1
        return res
```



##### C++ 算法实现

```

```

### 第十四题 最长的公共子串

#### 算法描述

1. 枚举的思想,判断每个字符串的第一位是否一致 如果一致就进行第二位的判断 如果连第一位都不一致 那就直接空

#### 算法实现

##### Python 算法实现

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:return strs
        i = 0
        res = ""
        while  i < len(strs[0]):
            _ = strs[0][i]
            for  data in strs:
                if i >= len(data) or data[i] != _  :
                    return res
            res += _
            i += 1
        return res
```



##### C++ 算法实现

```c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        string res;
        if (strs.size() == 0) return res;    // 特判传入参数为 空字符串

        for(int i = 0;;i++){
            if (i >= strs[0].size()) return res;   // 以 数组第一个为枚举对象 判断其他的字符串是否都有这个字符
            char _ = strs[0][i];                       // 待检查字符
            for(auto & str : strs)
                if (str[i] != _ || i >= str.size())
                    return res;
            res += _;
        }
        return res;
    }
};
```

### 第十五题  三数之和 

> 相向双指针 题二

#### 算法描述

1. 起初打算暴力来解决问题,两个指针指向的数据先求和 在取反 判断取反后的数据是否存在nums中,存在 则找到一组三元组并记录三个元素的下标 _i ndex 以及 取反后的数据下标 _append   超时  时间复杂度 O(n²)

2. 先对nums 进行排序, 三个指针 i j k  i<j<k 的规则,当 i 固定后 需要满足 三个指针相加为0  且在排序后的基础上  j 指针向后移动 k指针必定需要向前移动才可以满足三个相加为0 

    

#### 算法实现

##### Python 算法实现

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        _len = len(nums)
        _res = []
        if(not nums or _len<3):  return _res
        nums.sort()

        for index in range(_len - 2):
            # 若 nums[i]>0:因为已经排序好,所以后面不可能有三个数加和等于 0,直接返回结果.
            if(nums[index] > 0):  
                return _res
            # 如果当前值 和最大值 次大值相加依旧小于0  说明当前这个值可以跳过,以为下一个值可能会比当前这个值大
            if (nums[index] + nums[-2] + nums[-1] < 0): continue
            # 因为已经排序好,会出现相同值的情况,当前索引与上一个索引值一样,就不需要在进行判断一次,目的去除重复解.
            if(index > 0 and nums[index]==nums[index - 1]):
                continue
            L = index + 1
            R = _len - 1
            while(L < R):
                if(nums[index] + nums[L] + nums[R] == 0):
                    _res.append([nums[index],nums[L],nums[R]])
                    # 第二波 去除重复解 
                    while(L < R and nums[L] == nums[L+1]):
                        L += 1
                    while(L < R and nums[R] == nums[R-1]):
                        R -= 1
                    L += 1
                    R -= 1
                elif(nums[index] + nums[L] + nums[R] > 0):
                    R -= 1
                else:
                    L += 1
        return _res
```



##### Python 暴力算法实现  308测试集 超时...

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        _list = []
        _set = set()
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                k = ~(nums[i] + nums[j]) + 1
                if k in nums :
                    _k = nums.index(k)
                    if i!=j and  i!= _k and j != _k and nums[i] +nums[j] + k ==0:
                        _data = [nums[i],nums[j],k]
                        _data.sort()
                        if str(_data) not in _set:
                            _set.add(str(_data))
                            _list.append(_data)

        
  
        return _list
```



### 第十六题 最接近的三数之和

#### 算法描述

1. 与上题类似,一个索引index,双指针算法

2. 先对nums 进行升序排序
3. 特判
    +  如果target 大于等于排序后 最后三个之和 就直接返回最后三个之和 
    + target 小于等于 排序后前三个之和 类似
4. index 作为 第一个数,负责从后往前计算,由于题干明确不会出现相同解,所有不需要上题类似的检查重复解
5. L 左指针 R 右指针  计算出 三个数之和 与 target的大小  再计算出二者之间 数轴上的距离
6. 如果出现了当前距离大于新计算的距离 说明出现了更符合条件的三数之和, 更新距离值 三数之和 
7. 更新过后需要继续检查是否存在更好的解，需要判断三数之和 和 target的大小关系 进行 L++ 或 R -- 的操作

#### Python算法实现

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        _len = len(nums)
        _min_length = 10**7
        # _res = []   # 存放 返回结果的 三个元素
        nums.sort()
        _ans = 0
        # 特判两种情况 
        if target >= sum(nums[-3:]): return sum(nums[-3:])
        if target <= sum(nums[:3]): return sum(nums[:3])
        for index in range(_len):
            L = index + 1
            R = _len - 1
            while(L < R):
                # 计算 三数之和 与 target的差值
                _sum = nums[index] + nums[L] + nums[R]
                # _max _min 代表三数之和 和 target中 在数轴上,谁更靠右(大)谁更靠左(小) 
                _max = _sum if _sum > target else target
                _min = _sum + target - _max
                if(_max == _min): return _sum
                # 判断 是否需要更新 三数之和 和 target 数轴上的最小距离
                if (_min_length > _max - _min):
                    _min_length = _max - _min
                    _ans = _sum
                    # _res = [nums[index] , nums[L] , nums[R]]
                if(_sum > target):
                    R -= 1
                else:
                    L += 1
        return _ans
```



### 第十七题  电话号码的字母组合

#### 算法描述

1. 建立映射表
2. 对上一个元素列表中所有值都进行当前列表的 依次追加

#### Python 算法实现

##### 非递归算法

```Python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        _dict = {"2":["a","b","c"],"3":["d","e","f"],"4":["g","h","i"],"5":["j","k","l"],"6":["m","n","o"],"7":["p","q","r","s"],"8":["t","u","v"],"9":["w","x","y","z"]}
        if not digits: return []
        digits_list = list(digits)
        _sum = _dict[digits_list[0]]
        _init = []
        for index in range(1,len(digits_list)):
            for i in range(len(_sum)):
                for j in _dict[digits_list[index]]:
                    _init.append(_sum[i]  + j)
            _sum = _init[:]
            _init = []
        
        return _sum
```



##### 递归算法

```python


            
```

### 第十八题 四数之和

#### 算法描述

1. 与 三数之和 很类似
2. 四数之和 需要 两个固定的 一个 index  一个 j = index + 1    两个指针 L  R   

#### Python 算法实现

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        _len = len(nums)
        _res = []
        for index in range(_len):
            # 第一波结果去重
            if (index  and nums[index] == nums[index - 1]): continue
            for j in range(index + 1,_len):
                # 第二波结果去重
                if (j > index + 1 and nums[j] == nums[j - 1]): continue
                L = j + 1
                R = _len - 1
                while(L < R):
                    _sum = nums[index] + nums[j]+ nums[L]+ nums[R]
                    if (_sum == target):
                        _res.append([nums[index] , nums[j] , nums[L] , nums[R]])
                        while (L < R and nums[L] == nums[L + 1]): L += 1
                        while(L < R and nums[R] == nums[R - 1]): R -= 1
                        L += 1
                        R -= 1
                    elif _sum > target:
                        R -= 1
                    else:
                        L += 1


        return _res
```

### 第十九题 删除链表倒数第N个结点

#### 算法描述

感慨万千,曾经面试中灵机应变可以想出来的题.现在却想不起来...

1. 初始化几个指针  左 右 头结点
2. 先让右指针移动N次, 然后 左指针和 右指针 一起向后移动,直到右指针的next 为空.此时左指针对应的位置就是待删除的结点 

#### Python 算法实现

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        left = right = dummy = ListNode(next=head)
        # 没事先走几步
        for _ in range(n):
            right = right.next
        while right.next:
            left = left.next
            right = right.next
        left.next = left.next.next
        return dummy.next

```



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

### 第二十一题  合并两个有序链表

#### 算法描述

1. 因为已经有序 直接可以比较两个指针所指向的值的大小关系,、
2. 每次会比较出一个较小值 然后可以将这个值进行追加到头结点上,然后 较小值所在的链表的指针往后移动
3. 直到出现一条链表结束 然后 将链表尾指针指向剩下另一条链表
    + 当有一个链表为空的时候 就把另一条链表接在尾指针后面

#### Python 算法实现

```Python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1: return list2
        if not list2: return list1
        dummy =  curr = ListNode()
        
        while (list1 and list2):
            if list1.val <= list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next
        
        curr.next = list1 if list1 else list2
        return dummy.next
```



### 第 二十二题  括号生成

#### 算法描述

1. 关于括号的一些描述
    + 1. 在任意前缀中 “(” 的个数 >= “)” 的个数
        2. 在任意前缀中 左右括号个数相等
2. 每组合法的括号都有 2*n  位,可以一位一位分析
    + 当 左括号个数 <n  可以填充一个左括号
    + 当 右括号<n 且 左括号的个数大于右括号 可以填充右括号

#### Python 递归算法实现

```python
class Solution:
    # 任意子缀中 左括号的个数 大于等于 右括号的个数
    _res = []
    def dfs(self,_len,left_c,right_c,_seq):
        if left_c == _len and right_c == _len:
            Solution._res.append(_seq)
        else:
            if left_c < _len:
                self.dfs(_len,left_c + 1,right_c,_seq + "(")
            if right_c < _len and left_c > right_c:
                self.dfs(_len,left_c,right_c + 1,_seq + ")")

    def generateParenthesis(self, n: int) -> List[str]:
        Solution._res = []
        self.dfs(n,0,0,"")
        return Solution._res
```



### 第二十三题  [合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

#### 算法描述

1. 实现一  分治算法 
    + 1. 先创建一个res的结点， 每次合并 res 和数组中新的数组
    + 2. 合并算法就是 第二十一题类似
2. 实现二 归并算法
3. 实现三 最小堆
    + 1. 

#### Python 分治算法实现

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def mergeTwoList(self,l1,l2): # 第二十一题类似
        move = dummy = ListNode()
        while (l1 and l2):
            if l1.val < l2.val:
                move.next = l1
                l1 = l1.next
            else:
                move.next = l2
                l2 = l2.next
            move = move.next
        move.next = l1 if l1 else l2
        return dummy.next

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        _res = None      # 关键一
        if not lists:return _res
        for list_i in lists:
            _res = self.mergeTwoList(_res,list_i)   # 关键二
        return _res
```



#### Python 归并算法实现



#### Python 最小堆算法实现

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq        # 调用堆
        minHeap = []        # 存放堆
        # 建堆
        for listi in lists: 
            while listi:
                heapq.heappush(minHeap, listi.val)   # 把listi中的数据逐个加到堆中
                listi = listi.next
        dummy = ListNode(0) #构造虚节点
        move = dummy
        while minHeap:
            move.next = ListNode(heapq.heappop(minHeap)) #依次弹出最小堆的数据
            move = move.next
        return dummy.next 
```

#### Python 数组实现

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 创建一个空的头节点
        dummy = ListNode(None)
        # 创建一个指针指向头节点
        cur = dummy
        # 创建一个存储结点的列表
        node_list = []
        # 遍历列表中的每一个链表
        for l in lists:
            # 遍历每一个链表
            while l:
                # 将每一个结点都放入node_list中
                node_list.append(l)
                # 继续遍历每一个链表
                l = l.next
        # 将node_list按照结点值从小到大排序
        node_list.sort(key=lambda x:x.val)      # <-- 闪光点！！！！
        # 遍历node_list
        for node in node_list:
            # 将排序后的结点放入dummy头节点的后面，并更新cur指针
            cur.next = node
            cur = cur.next
        # 返回dummy头节点的下一个结点
        return dummy.next
```

### 第二十四题 [两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

#### 算法描述

0. 本题发现,头结点会发生变化,可以新建立一个虚拟头结点

1. 拥有头结点move指针以及 相邻的两个指针 left 和 right ,三个指针,move指针指向要交换的两个结点的前一个结点
1. 交换结束后move指针变成交换前 left指针指向的数

#### Python算法实现

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = move = ListNode(next = head)    # 建立虚拟头结点并指向head
        while move.next and move.next.next:
            left = move.next
            right = move.next.next
            # 开始交换结点！
            move.next = right
            left.next = right.next
            right.next = left
            # 开始进行下两个结点的交换 left 为 交换后 更靠近下一个结点的指针
            move = left 
        return dummy.next
```

#### Python 递归算法实现   暂未明白 ~

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head==None or head.next==None:
            return head
        next = head.next
        head.next = self.swapPairs(next.next)
        next.next = head
        return next
```



### 第二十五题   [ K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

#### 算法描述

1. 与上题类似,需要虚拟一个头结点 dummy = move 
2. 首先去判断从 move结点开始计算 是否存在K个元素 存在下一步 不存在就返回
    + 1. 将 K 个中 除两边元素之外所有元素的指针调转方向
    + 2. 



#### Python 算法实现 - 数组存放K个指针方案

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = move = p = ListNode(next = head)
        # p 是向前确定k个元素的指针 move 是k个元素为一组的前一个指针
        cnt = 0
        while p.next:
            _list = []
            while cnt < k and p.next:
                _ = p.next
                _list.append(_)
                p = p.next
                cnt += 1
            if cnt == k:
                # 交换k组中 第一个 和最后一个元素的指针  _list[0].next = _list[-1].next
                move.next = _list[-1]
                _list[0].next = _list[-1].next
                cnt -= 1
                while cnt:
                    _list[cnt].next = _list[cnt - 1]
                    cnt -= 1
                p = move = _list[0]
            else:
                # cnt 不足k个 返回
                break
        return dummy.next
```

#### Python 算法实现 高效算法  灵神视频

```python

```



### 第二十六题 [删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

####  算法描述

双指针 pre 指向真正的待填充数据, suf 指向检查字符串是否与 pre指针相同 

#### Python 算法实现

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        pre = suf = 0
        while suf < len(nums):
            if nums[suf] != nums[pre]:
                pre += 1
                nums[pre] = nums[suf]
            suf += 1
        
        return pre + 1
```



### 第二十七题  [移除元素](https://leetcode.cn/problems/remove-element/)

#### 算法描述 

与上体类似 不过 suf 与 val比较而不是与 pre比较

#### Python 算法实现

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        while(val in nums):
            nums.pop(nums.index(val))
        return len(nums)
```



### 第二十八题 [找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

#### 算法描述

KMP 算法吧

利用Python特性进行解题

#### Python 算法实现

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle in haystack:
            return haystack.index(needle)
        return -1
```

#### Python KMP 算法实现 待续

```python

```



### 第二十九题 [ 两数相除    未解决](https://leetcode.cn/problems/divide-two-integers/)

#### 算法描述

1. 数学题 

#### Python 算法实现

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        is_minus = False
        #
        if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0): is_minus = True 
        dividend, divisor = abs(dividend),abs(divisor)
        ext = [ pow(2,i) * divisor for i in range(31)]
        _res = 0
        for i in range(len(ext)-1,-1,-1):
            if dividend >= ext[i]:
                dividend -= ext[i]
                _res += 1 << i
        if is_minus: return -_res
        return res if -2**31 <= res <= 2**31-1 else 2**31-1
    
    
    

class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        # x << i, 左移相当于 x* 2^i
        # x >> i, 右移相当于 x/ 2^i
        if abs(dividend) < abs(divisor): return 0
        limit = 2**31 - 1
        neg = (dividend <0) != (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res, div, temp = 0, divisor, 1
        while dividend >= divisor:
            while dividend > (div << 1):
                div <<= 1
                temp <<= 1
            dividend -= div
            div = divisor
            res += temp
            temp = 1
        res = -res if neg else res
        return res if -2**31 <= res <= 2**31-1 else limit
```



### 第三十题 [串联所有单词的子串](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/)

#### 算法描述

1. 滑动窗口  枚举 实现
2. 将 words 中的元素 放入dict中 初始化每个出现次数都为1 
3. 枚举从 0 到 len(s) - len(words[0]) + 1  ，将 整个s 分为len(s) // len(words[0]) 段来处理
4. 每段又分为 len(words[0]) * len(words) 个小组,而每个小组为肯定出现在dict中 
5. 最后统计dict中value 为0  则记录索引 不为0  开始下一处索引

#### Python 算法实现   暂时不优化了~~

```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s :return []
        _len, w_list_len, w_str_len, _res = len(s), len(words),len(words[0]), []
        words_dict = {}
        start = 0 
        for _ in words:
            words_dict[_] = words_dict.get(_,0) + 1
        for start in range(_len - (w_list_len * w_str_len) + 1):
        # while start < _len - (w_list_len * w_str_len) + 1:
            # 每次都重新获取一个备份
            check_words_dict = words_dict.copy()
            # 将s 按照 串联子串长度 分为一段一段 
            check_strs = s[start:start + w_list_len * w_str_len]
            # 检查每段中的每组(长度为 w_str_len) 是否存在 dict中
            check_index = 0
            while check_strs[check_index:check_index + w_str_len] in check_words_dict and check_words_dict[check_strs[check_index:check_index + w_str_len]] > 0:
                check_words_dict[check_strs[check_index:check_index + w_str_len]] -= 1
                # 更新检查索引位置
                check_index += w_str_len
            if sum(check_words_dict.values()) == 0:
                _res.append(start)
            #     start += w_str_len
            # else:
            #     start = start + 1
        return _res
```



### 第三十一题  [下一个排列](https://leetcode.cn/problems/next-permutation/)

#### 算法描述

1. 又是思维题目, 题目描述为 在给定的数组中,寻找一个非降序子数组 [a,b] (b是b之后的元素中最大的元素，a是比b小的相邻元素  ),然后再其后找一个大于第一个元素的最小元素 调换二者位置,然后将后面重新按照升序排列即可
2. 1. 首先 先找到第一个开始降序的位置

#### Python算法实现

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        # 1. 先找到一个开始降序的位置
        _len, start, k = len(nums), -1, len(nums) - 1
        # 从后往前找 第一组升序的位置 k为最大的元素,也就是第k个元素比第k-1 个元素要大 
        # 如 1,3,5,4,2   需要找的就是 元素5 索引k = 2 为  
        while(k > 0 and nums[k - 1] >= nums[k]): k -= 1
        # 特殊情况 遇到是降序的数组 直接颠倒所有元素即可
        # 即从尾到头 都找不到一个满足 nums[k]> nums[k-1 ] 的元素
        if (k <= 0 ): 
            nums.reverse()
        else:
            index = k
            # 找 从k之后 第一个 小于等于k-1的 元素 索引位置 index 那么 index -1 就是 大于第k-1 个元素的最小元素
            while(index < _len and nums[index] > nums[ k - 1]): index +=1
            nums[index - 1 ],nums[k - 1] = nums[k - 1],nums[index - 1]
            # 从 最大元素 位置开始 到 数组结束 进行 升序 
            start = k
            end = _len -1
            while start < end:
                nums[start],nums[end] = nums[end],nums[start]
                start += 1
                end -= 1
```

#### 优化之后算法

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        # 1. 先找到一个开始降序的位置
        _len, start, k = len(nums), -1, len(nums) - 1
        # 从后往前找 第一组升序的位置 k为最大的元素,也就是第k个元素比第k-1 个元素要大 
        # 如 1,3,5,4,2   需要找的就是 元素5 索引k = 2 为  
        # while(k > 0 and nums[k - 1] >= nums[k]): k -= 1
        for k in range(_len - 1,-1,-1):
            if nums[k] <= nums[k- 1]:
                continue
            else:
                # 找到了k 索引
                # 特殊情况 遇到是降序的数组 直接颠倒所有元素即可
                # 即从尾到头 都找不到一个满足 nums[k]> nums[k-1 ] 的元素
                if (k <= 0 ): 
                    nums.reverse()
                    break
                else:
                    # 找 从k之后 第一个 小于等于k-1的 元素 索引位置 index 那么 index -1 就是 大于第k-1 个元素的最小元素
                    index = k
                    while(index < _len and nums[index] > nums[ k - 1]): index +=1
                    nums[index - 1 ],nums[k - 1] = nums[k - 1],nums[index - 1]
                    # 从 最大元素 位置开始 到 数组结束 进行 升序 
                    end = _len -1
                    while k < end:
                        nums[k],nums[end] = nums[end],nums[k]
                        k += 1
                        end -= 1
                    break
```



### 第三十二题   [最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)   竞赛题 难

#### 算法描述

1. 括号题的两个重要性质
    + 其一, 任何合法的子缀中 左右括号数量相等
    + 其二, 合法的子缀中,左括号的数量大于等于右括号的数量
2. 我们可以先找到第一处括号不合法的位置,然后将其断开.这是因为在此处之前已经不合法了,合法的不会横跨这一段
3. 依次划分出所有的不合法的位置 然后将其拆分开来,其中每一段除最后一个右括号以外都满足左括号个数大于等于右括号个数 也就是除了这个右括号以外,在此之前的所有右括号都可以找到对应的左括号
4. 有3可以得知,当遇到一个右括号且栈 没有左括号的时候就说明这段已经结束了需要从下一段寻找
5. 

#### Python 算法实现

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # start 为有效括号开始索引的前一个位置记录
        # 也可理解为 start为 遇到最后一个右括号时,与之对应的左括号的前一个位置 也就是每一段的结尾
        # 若 start 设为开始索引 后面需要减去start后 再多加1的操作 麻烦
        start = -1
        stack = []
        _res = 0
        for i in range(len(s)):
            if s[i] == '(':
                # 若为左括号 将下标存入栈内
                stack.append(i)
            else:
                # 遇到右括号检查栈是否有左括号位置
                if len(stack):
                    stack.pop()
                    # 再次检查是否还有左括号，如果还有,说明必有一个右括号等着
                    if len(stack):
                        _res = max(_res,i - stack[-1])
                    else:
                        # 说明当前这个右括号与之匹配的已经消除了
                        _res = max(_res,i - start)
                else:
                    # 遇到右括号发现栈空 说明是 违规的字符串 重新记录start
                    # 遇到了 右括号数量大于左括号数量时的右括号索引
                    start = i
        return _res
```



### 第三十三题   [搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

#### 算法描述

1. 多段升序的数组,数组其中一段满足所有的值都大于nums[0] 另一端所有元素都满足小于nums[len(nums) - 1] 这样的特性, 找出这个分界点 依次使用两次二次进行
2. 

#### Python 算法实现





### 第 三十四题       [在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

> 二分查找 实现 [] [)   (]  以及 >=    >    <=   <  实现

#### 算法描述

#### Python 实现二分

1. 实现 左闭右闭区间 二分查找算法

    + ```python
        def lower_bound(nums: List[int], target: int) -> int:
            left = 0
            right = len(nums) - 1
            while left <=  right:    # 闭区间 [left,right]
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left
        ```

2. 实现 左闭右开区间 二分查找算法

    + ```python
        def lower_bound2(nums: List[int], target: int) -> int:
            left = 0
            right = len(nums)
            while left <  right:    # 闭区间 [left,right)
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left
        ```

3. 实现 开区间 二分查找算法

    + ```python
        def lower_bound3(nums: List[int], target: int) -> int:
            left = -1
            right = len(nums)
            while left + 1 <  right:    # 开区间 (left,right)
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid
                else:
                    right = mid
            return right
        ```

4. 实现 大于 小于 小于等于 大于等于

    + ```python
        # 有序数组上的二分查找分为4中,≥ > ≤ <  可以相互转换
        # >x : 可以看为 ≥(x + 1)
        # <x : 可以看为 ≥(x) - 1
        # ≤ : 可以看完 (>x) - 1
        ```

        

#### Python 算法实现

```python
 def lower_bound(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1
    while left <=  right:    # 闭区间 [left,right]
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        start = lower_bound(nums,target)
        if start == len(nums) or nums[start] != target:return [-1,-1]
        # 求最后一次出现target的位置 等价于 求 >=target+1 第一次出现的位置 再减1 
        end = lower_bound(nums,target + 1) -1
        return [start,end]
```



#### Python 耍贱的实现

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if target in nums:
            return [nums.index(target),nums.index(target)+ nums.count(target) - 1]
        return [-1,-1]
```

### 第三十五题

#### 算法描述

#### Python 算法实现



### 第三十六题   [有效的数独](https://leetcode.cn/problems/valid-sudoku/)

#### 算法描述

1. ·用长度为9 的列表来充当1-9,若出现6 就在索引为6处 设为True 若在一行中出现了索引为true的说明改行直接出现了重复数字
2. 列也是 

#### Python 算法实现

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 用来保存 1-9 是否存在过
        _list = []

        # 检查行 
        for row in range(9):
            _list = [False] * 9
            for cow in range(9):
                if  board[row][cow] != '.':
                    # 减 1  是因为 索引是从 0 开始的  所以向前移一位
                    _ = int(board[row][cow]) - 1
                    if _list[_]:return False
                    _list[_] = True
        
        # 检查列
        for row in range(9):
            _list = [False] * 9
            for cow in range(9):
                # 此处是 列 行  
                if  board[cow][row] != '.':
                    _ = int(board[cow][row]) - 1
                    if _list[_]:return False
                    _list[_] = True
        
        # 检查小方块
        for row in range(0,9,3):
            for cow in range(0,9,3):
                _list = [False] * 9
                for x in range(3):
                    for y in range(3):
                        if board[row + x][cow + y] != '.':
                            _ = int(board[row + x][cow + y])  - 1
                            if _list[_]: return False
                            _list[_] = True
        
        return True
```



### 第三十七题    [解数独](https://leetcode.cn/problems/sudoku-solver/)   困难

#### 算法描述



#### Python 算法实现

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        def dfs(board, x, y):
            # 先移动纵坐标
            if y >= 9:
                x += 1
                y = 0
            if x >= 9:
                return True
            if board[x][y] != '.': 
                return dfs(board, x, y + 1)
            else:
                for i in range(9):
                    # 先判断 要枚举的数 是否已经存在 列 行 宫 里面了
                    if (not row[x][i]) and (not cow[y][i]) and (not cell[x//3][y//3][i]):
                        board[x][y] = str(1 + i)
                        # print(board[x][y])
                        # 设为True 表示 对应的第i+1 个数 填充进去了
                        row[x][i] = cow[y][i] = cell[x//3][y//3][i] = True
                        if dfs(board, x, y + 1):
                            return True
                        else:
                            board[x][y] = '.'
                            # 填充失败 继续下一步
                            row[x][i] = cow[y][i] = cell[x//3][y//3][i] =  False
            return False
        """
        Do not return anything, modify board in-place instead.
        """
        # 保存 每行 每列 每宫1-9 每个数字 用来后续判断某个数是否已经使用过,不可以重复使用
        row = [[0 for i in range(9)] for j in range(9)]
        cow = [[0 for i in range(9)] for j in range(9)]
        cell = [[[0 for i in range(9)] for j in range(3)] for k in range(3)]

        # 将 board 保存进
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    # 转换为0-8 对应列表的索引为 1-9
                    _ = int(board[i][j]) - 1
                    row[i][_] = cow[j][_] = cell[i//3][j//3][_] =  True
        
        # dfs
        dfs(board,0,0)
```



### 第三十八题



### 第三十九题





### 第四十题





### 第四十一题





### 第四十二题 困难版 接最多雨水问题

> 接雨水问题 题二

#### 算法描述

1.  算出每个水桶左右两侧最大值 然后计算出结果

#### Python 算法实现

```python
class Solution:
    def trap(self, height: List[int]) -> int:

        # 找出每个方块左侧最大值 和 右侧最大值 
        # 计算出每个方块上可以盛多少水 = min(左侧最大值,右侧最大值) - 方块高度
        _res = 0
        _len = len(height)
        pre_max = [0] * _len
        # 对于左边最大高度数组来讲 第一个元素没有左边最大高度 不如初始化为本身
        pre_max[0] = height[0]
        for i in range(1,_len):
            pre_max[i] = max(pre_max[i - 1],height[i])
        
        suf_max = [0] * _len
        # 右侧最大高度 同理 
        suf_max[-1] = height[-1]
        # 数组本身从 _len - 1开始 每次 -1 但因为第一个数已经填充了 就从倒数第二个开始
        for i  in range(_len-2,-1,-1):
            suf_max[i] = max(suf_max[i + 1],height[i])
        
        for h,pre,suf in zip(height,pre_max,suf_max):
            _res += min(pre,suf)  - h
        return _res
```

#### Python算法实现 未完成

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        _res = 0
        _len = len(height)
        left = 0
        while (left < _len):
            right = left + 1
            while (right < _len and height[right] < height[left]): right += 1
            if right < _len and right - left - 1:
                _all = (right - left - 1)* min(height[right] , height[left])
                _close = sum(height[left + 1 : right])	
                _res += _all - _close
                left = right
            else:
                left += 1
        return _res
```



### 第六十九题 [ x 的平方根 ](https://leetcode.cn/problems/sqrtx/)  

#### 算法描述

1. 题目可以翻译为  从0 开始 到 x ,中间必定有一个数的平方是 x 或 平方小于等于x
2. 换句话来说  找 0 到x 中  mid*mid 小于等于 x  直接模板

#### Python 算法实现

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        # 类似于二分找 某个值的平方 <= x
        #  小于等于   模板一 情形三
        left, right = 0,x
        while left <= right:
            mid = left + (right - left) // 2
            if mid * mid <= x:
                left = mid + 1
            else:
                right = mid - 1
            
        return right
```

### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode.cn/problems/que-shi-de-shu-zi-lcof/)

#### 算法描述

1. 索引从0 开始 判断 第 i个索引的位置是会否为i
2. 如果是i 说明 i的左侧满足条件 需要后移动
3. 如果 大于i 说明 i的左侧发生的变化 目标值在左边 移动 right

#### Python 算法实现

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # 索引从0 开始 判断 第 i个索引的位置是会否为i
        # 如果是i 说明 i的左侧满足条件 需要后移动
        # 如果 大于i 说明 i的左侧发生的变化 目标值在左边 移动 right
        left, right = 0, len(nums) -1 
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == mid:
                left = mid + 1
            elif nums[mid] > mid:
                right = mid - 1
        return left
```

### 第八十一题 [搜索旋转排序数组 II](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)

#### Python算法实现

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        # 先二分 找中间断点
        # 断点使 左侧到断点处 左闭右闭都是不降序 断点到数组结尾 ‘’左‘’开右闭区间 也是不降速
        # 测试集虽然过了 但不是真真意义上的 先找二分找断点 再二分找答案。。。
        left,right = 0,len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target or nums[right] == target:
                return True
            elif nums[mid] < nums[right]:
                right = mid
            elif nums[mid] > nums[right]:
                left = mid + 1
            else:
                right -= 1
        # 运行到此处,若没有返回True,但找到了断点   right,nums[right]
        # return right,nums[right]
        if nums[0] < target:
            left = 0
        elif nums[0] > target:
            right = len(nums) - 1
        else:
            return True

        # 已确定target大致区间  在一段不降序区间 开始普通二分查找
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return False if left == len(nums) or nums[left] != target else True
```



### 第一百五十三题  [寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

#### 算法描述

1. 取数组中最后一个元素，这个元素只有两种性质,要么是最小值,要么位置在最小值的右侧
2. 二分数组从 0 到 n-2 开始 若 mid 大于 最后一个元素 说明最小值一定在mid的左侧 反之在mid右侧
3.  

#### Python 算法实现

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = -1
        right = len(nums) - 1
        while left + 1< right :
            mid  = (left + right) // 2
            if nums[mid] > nums[-1]:
                left = mid
            else: 
                right = mid 
        return nums[right]
```

#### Python 算法实现2

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # 相当于 33题 先找到断点 也就是 nums中最大值
        # 当退出循环时,left 指向 满足 大于第一个元素的最大元素的下一位
        # right 为最大元素的位置
        # 判断 right 是不是最后一位,是说明整个nums是升序的 最小的是 nums[0]
        left , right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[0]:
                right = mid - 1
            else:
                left = mid + 1
        return nums[left] if right != len(nums) - 1 else nums[0]
```



### 第一百五十四题 [寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/)

#### 算法描述

...

#### Python 算法实现

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # 旋转数组 也就三种情况  先升序小部分,然后 剩下的所有元素都小于nums[0]
        # 升序一大部分, 剩下的小部分元素的最后一个 nums[len(nums) - 1]小于 num[0]
        # 再 就是 一直升序状态 最小值为 nums[0] 
        left , right = 0 , len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            # 因为原数组是 升序的！！！！ 可以保证  right的左边 小于等于 right
            if nums[mid] < nums[right]:
                right = mid
            elif nums[mid] > nums[right]:
                left = mid + 1
            else:
                # 此时 mid 和 right 指向的数据都一样  那说明 right 和 mid 是 重复值 
                # 删除去right 也无妨 ~
                right -= 1
        # 退出循环时
        # 1. 有 一直升序的 nums[0] 为最小值, left为答案
        # 2. 有 一部分升序 在升序的  最小值在 断点处下一个位置    left 还是答案
        # 3. 全为降序 最后一个元素为最小值 left为答案
        return nums[left]
```



### 第一百六十二题  [寻找峰值](https://leetcode.cn/problems/find-peak-element/)

> 二分查找 题二 思维跨度大 难理解

#### 算法描述

1. 为什么采用二分的开区间算法呢？
2. 

#### Python 算法实现

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        # [0,n-2]
        # [-1,n-1]
        left ,right= -1 ,len(nums) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid
            else:
                right = mid
        return right
```

### 第二百七十八题 [第一个错误的版本](https://leetcode.cn/problems/first-bad-version/)

#### Python算法实现

```python
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left , right = 1, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid 
            else:
                left = mid + 1
        return left
```



### 第六七四题 最长连续递增序列

#### 算法描述

dp思想来解题就会很快,但还不会

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

### 第一六七题 两数之和 || 

> 相向双指针 题一

#### 算法描述

1. 借助题干条件,数组已经先按照非递减的顺序排序,
2. 左指针指向第一个数 右指针指向最后一个数
    + 若 两个指针指向的数 == target 返回指针位置
    + 若 两指针之和大于target 说明右指针指向的数大需要向前移动来获取一个次大值
    + 若 之和小于target 说明 左指针 需要向后移动来获取下一个较大值

#### Python 算法实现

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        L = 0
        R = len(numbers) - 1
        while (L < R):
            _sum = numbers[L] + numbers[R]
            if _sum == target:
                return [L + 1,R + 1]
            elif _sum > target:
                R -= 1
            else:
                L += 1
```





### 第一零三一题 两个非重叠子数组最大和 每日一题  没做出来

#### 算法描述

#### 算法实现

### 第二零九题 长度最小的子数组    

> 同向双指针  滑动窗口   题一

#### 算法描述

1. 同向双指针 left right 初始都是0 指向同一个元素
2. 当 两个指针所包含的所有元素相加 小于 target 时, right + 1 
3. 若 累加和 大于等于 target 说明 已枚举的元素中，已经满足了条件,但需要继续执行，来检查最小长度
    + 满足条件 sum -= 左指针指向的数据 并 更新长度 以及 左指针后移一步  顺序不能颠倒

#### Python 算法实现

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        _len = len(nums)
        _sum = left = 0
        _res = 10**5 + 1 
        if sum(nums) < target: return 0
        if sum(nums) == target: return _len
        for right,num in enumerate(nums):
            _sum += num
            while (_sum >= target):
                _sum -= nums[left]
                _res = min(_res,right - left + 1)
                left += 1
        return _res
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

### 第七一三题 乘积小于K 的所有子数组

> 同向双指针  滑动窗口   题二

#### 算法描述 

1. 算法整体思想与 题一 类似
2. left right 指针指向的数据累乘若小于k 就将 两个指针的数据copy到新的数组
3. 若大于k  说明 两个指针之间的数据不满足条件了以及之后也更不可能满足条件(数组中每个元素都大于等于1)
4. 枚举  以右指针为准则 左指针为向后移动

#### Python 算法实现

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1: return 0
        _product = 1
        left = 0
        _res = 0
        for right,num in enumerate(nums):
            _product *= num
            while _product >= k :
                _product //= nums[left]
                left += 1
            _res += (right - left + 1)
        return _res
```







