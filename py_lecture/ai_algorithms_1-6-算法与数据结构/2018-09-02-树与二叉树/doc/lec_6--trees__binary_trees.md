
# 6. 树与二叉树

    1.什么是树;（了解）
    2.	树的应用;（了解）
    3.树的操作;（了解）
    4.	什么是二叉树;(掌握)
    5. 二叉树的性质(了解)
    6.二叉树的实现(掌握)

## 树的含义

树(tree),是一种数据结构，它是由$n (0 < n <\infty)$个结点组成一个具有层次关系的集合($T$).
对任意一棵树.
1. 有且只有一个根节点
2. 子树.

## 关于树的术语



#### 结点： 数据元素和若干指向子树的分支。
#### 树：
#### 森林：




### 树的特征：
结点数目： $n $有限整数; $n>0$.

    每个结点有0个或多个子结点;
    根：无父结点的结点;
    除根外，每个结点只有一个父结点;
    每个子结点可分为多个不相交的子树.

### 树的应用


## 二叉树

### 二叉树含义：

### 二叉树的形态： 
    5种
    
### 二叉树的性质：

### 特殊形式的二叉树



```python
class Node(object):
    #构造结点
    def __init__(self,item):
        self.elem = item  # 数据元素被保存在self.elem中
        self.lchild = None
        self.rchild = None

#实现完全二叉树
class Tree(object):
    #完全二叉树
    def __init__(self):
        self.root = None  # 根结点

    def add_node(self,item):
        node = Node(item)
        if self.root is None:
            self.root = node
            return
        queue=[self.root]
        while queue:
            cur =queue.pop(0) # 从queue的头部读出
            if cur.lchild is None:
                cur.lchild = node
                return
            else:
                queue.append(cur.lchild) # 添加左子结点
            if cur.rchild is None:
                cur.rchild = node
                return
            else:
                queue.append(cur.rchild) #添加右子结点

    # 深度优先遍历
    # 前序遍历: TLR
    def preorder(self,root):
        if root == None:
            return
        print(root.elem)
        self.preorder(root.lchild)
        self.preorder(root.rchild)
        
    # 中序遍历： LTR
    def inorder(self, root):
        if root == None:
            return 
        self.inorder(root.lchild)
        print(root.elem)
        self.inorder(root.rchild)
        
    # 后序遍历 ：LRT
    def postorder(self,root):
        if root == None:
            return
        self.postorder(root.lchild)
        self.postorder(root.rchild)
        print(root.elem)
        
    # 广度优先遍历
    def breadth_travel(self):
        """广度优先遍历：用队列实现"""
        if self.root == None:
            return
        queue = [self.root]
        
        while queue:
            cur = queue.pop(0)
            print(cur.elem)
            #在访问当前结点时，把当前结点的左，右子结点加入到队列
            if cur.lchild != None:
                queue.append(cur.lchild)
            if cur.rchild != None:
                queue.append(cur.rchild)
    # 序列化            
    def tree2str(self,root):
        res = ""
        if root == None:
            return "#!"
        res += str(root.elem)  + "!"
        res += self.tree2str(root.lchild)
        res += self.tree2str(root.rchild)
        return res

    
# 把文件转换成二叉树
def str2tree(string):
    # 提取元素,保存到队列（queue）中 
    values = string.split("!")
    print(values)
    queue = []
    for x in values:
        queue.append(x)
    # 重构二叉树
    return recon(queue)


def recon(queue):
    """重构二叉树"""
    if queue != []:
        val = queue.pop(0)
        if val == '#' or val =='':
            return None
        node = Node(int(val))
        node.lchild = recon(queue)
        node.rchild = recon(queue)
        # 返回结点
        return node

if __name__ == "__main__":
    tree = Tree()
    tree.add_node(1)
    tree.add_node(2)
    tree.add_node(3)
    tree.add_node(4)
    tree.add_node(5)
    print("前序遍历：")
    tree.preorder(tree.root)
    print("中序遍历：")
    tree.inorder(tree.root)
    print("后序遍历：")
    tree.postorder(tree.root)
    print("广度优先遍历：")
    tree.breadth_travel()
    print("序列化：")
    s = tree.tree2str(tree.root)
    print("结果：",s)
    root = str2tree(s)
    print("前序遍历重构的二叉树：")
    tree.preorder(root)
    
```

    前序遍历：
    1
    2
    4
    5
    3
    中序遍历：
    4
    2
    5
    1
    3
    后序遍历：
    4
    5
    2
    3
    1
    广度优先遍历：
    1
    2
    3
    4
    5
    序列化：
    结果： 1!2!4!#!#!5!#!#!3!#!#!
    ['1', '2', '4', '#', '#', '5', '#', '#', '3', '#', '#', '']
    前序遍历重构的二叉树：
    1
    2
    4
    5
    3
    

## 二叉树的遍历

按一定规律访问二叉树上的每个结点,且每个结点**只能访问一次**，这种操作称为遍历（traversal）.

树的两种重要的遍历模式是**深度优先遍历和广度优先遍历**,深度优先一般用递归或栈，广度优先一般用队列.

##  二叉树的应用： 二叉树的序列化和反序列化
假设二叉树结点里存了整数。
12       “12！”
None     “#！”
“1！2！5！3！4！6！#！” （序列化的结果）



