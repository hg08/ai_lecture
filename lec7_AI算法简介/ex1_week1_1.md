## 习题一

**1**：(**30分**)两矢量点积可以用两矢量的长度，以及夹角的余弦表示为
$$
{\bf x} \cdot {\bf y} = |{\bf x}||{\bf y}|\text{cos}\theta. \nonumber
$$
已知
$$
{\bf a} = \left(
\begin{matrix}
4 \\
0
\end{matrix}
\right),
{\bf b} = \left(
\begin{matrix}
-4 \\
0
\end{matrix}
\right),
{\bf c} = \left(
\begin{matrix}
4 \\
3
\end{matrix}
\right). \nonumber
$$
求矢量**a**与矢量**c**的夹角；求矢量**b**与矢量**c**的夹角．用python代码实现这个功能．程序名取为"**dot_product.py**". 使得运行命令

　　　　**python dot_product.py**

能够正常运行．要求输出结果到名为"**output_\*\*.txt**"的文件中(\*\*为你的姓名，可以用拼音). 文件中包含两行内容，格式如下：



a与c夹角的余弦: \*\*\*

a与c的夹角(rad): \*\*\*

a与c的夹角(°)： \*\*\* 



说明：　

1. 有模板可用，下载地址: https://github.com/hg08/ai_lecture/blob/master/week1/dot_product.py
2. 模板不是必需的，你可以完全自己写．(以后省略此说明)



**2**：(**20分**)计算下列矩阵乘法．用Python代码实现计算．你需要写一个名为"**matrix_product.py**"的Python文件,运行命令

　　　　**python matrix_product.py**

依次输出(1)(2)两题的结果．格式如下：



AB: \*\*

Zc: \*\*



(1).(10分)
$$
AB=\left(
\begin{matrix}
0 &  1
\end{matrix}
\right) 
\left(
\begin{matrix}
0 \\
1
\end{matrix}
\right);  \nonumber
$$
(2)(10分)
$$
Z{\bf c}=\left(
\begin{matrix}
1 &  0\\
0 &  -1
\end{matrix}
\right) 
\left(
\begin{matrix}
0 \\
1
\end{matrix}
\right). \nonumber
$$
说明：代码有模板可用，下载地址:　https://github.com/hg08/ai_lecture/blob/master/week1/matrix_product.py　

3: (**30分**)

(1)(10分) 计算下面各式,
$$
-1\left(
\begin{matrix}
1\\
1 \\
0
\end{matrix}
\right) ;

-2\left(
\begin{matrix}
1\\
0 \\
1
\end{matrix}
\right) 

\nonumber
$$


(2) (10分) 并对计算结果再求出其模．

(3) (10分)试说明中常数为**-1**, -2时，数乘分别表示对矢量做什么操作？

**4**: (**20分**) 假设我们有一本词典,　里面只有五个单词 [a, b, c, d, e]. 这里有三个文档：

文档A: [a,b, b, d,d,e]

文档B: [b,b,b,e,e,e,d,a]

文档C: [d,b,b,e]

使用bag-of-words模型将每个文档表示成五维向量,每个维度上的分量分别代表a,b,c,d,e这五个单词在文档中出现的次数. 例如文档A可表示为
$$
{\bf a} = \left(
\begin{matrix}
1 \\
2 \\
0 \\
2 \\
1
\end{matrix}
\right). \nonumber
$$
(1)(8分)将文档B和文档C按照上述模型表示成矢量.

(2) (12分)若定义两矢量夹角的余弦值为这两个矢量所对应的文档的**相似性**．计算文档A与文档B的相似性，计算文档A与文档C的相似性.

模板地址：https://github.com/hg08/ai_lecture/blob/master/week1/cos_similarity.py