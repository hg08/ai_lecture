# 机器学习算法入门介绍

## 1.课前准备(1): 设置运行环境



1. 安装Anaconda环境： <https://www.anaconda.com/>

2. 确认在Anaconda中已经安装如下模块：numpy, pandas, mglearn，sci-kit learn

 （1）查看Anaconda中已经安装的模块

 ```pip list```

 或

 ```conda list```

（2）添加适当的channels. 如：打开Anaconda prompt, 在命令行输入:

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda config --set show_channel_urls yes
```

同理，对于不想要的源，也可以删除, 比如

```shell
conda config --remove channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'
```

（3）安装模块的方法：

打开Anaconda prompt终端, 在命令行输入:（以mglearn为例）

```shell
pip3 install mglearn
```

或者指定具体的源，如：

 ```shell
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple mglearn
 ```

又如，要安装PyTorch,可用如下方法：

``` 
conda install pytorch torchvision cudatoolkit=10.0
```

详情可参考 [1,2].



(4). 打开Jupyter Notebook的方法：

A. 在终端命令行打开：

```shell
jupyter notebook
```

B. 在anaconda环境里找到Jupyter,点击进入. 

(5). 共享Python环境之方法

在终端输入如下命令，以将当前的Conda环境全部导出为environment.yaml文件:

conda env export > environment.yaml

**注意：1. 本次课程主要包括几个简单的机器学习入门算法, 分为原理和与操作，大家可以带上电脑，便于练习；2. Anaconda环境的安装比较简单，我们这里搜集的习题都用Python实现. 建议大家先安装好Anaconda环境，以便可以实时运行示例和练习中的代码. **

 

## 2. 课前准备(2): 自学

（参考Wes McKinney, Python for Data Analysis,附录及相关章节 <https://github.com/wesm/pydata-book>）

(1) Python基础知识: 简单数据结构及操作.

(2) NumPy数组的基本操作. 

(3) 安装Anaconda环境，学习使用Jupyter. （练习：启动Anaconda，运行Jupyter notebook, 然后打开.ipynb格式的文件.）

 

## 3.课程内容介绍

### 1. 机器学习基本概念、k近邻算法介绍： (11月5日上午9：00--12：00)

(1) 数据集，训练，测试，样本，特征，标签等；机器学习的类别及具体实例：监督学习，非监督，强化学习；机器学习处理的两类问题：分类，回归.

(2) k近邻算法的含义和基本思想：距离函数，投票函数，kNN分类，kNN回归.

(3) 练习：词频统计，简单的k近邻分类.

### 2. 线性模型及应用 (11月5日下午 14：00--17：00)

基本概念：损失函数(Cost function)， 梯度下降法(Gradient Descent)，正则化

(1)单变量线性模型；多变量线性模型

(2)正则化线性模型(Regularized linear models):目的及方法. 具体的实现方式：岭回归，Lasso回归，Elastic Net回归.

(3)练习：分析正则化线性模型的结果，评估机器学习模型的质量.

 

### 3. 决策树算法 (11月6日上午 9：00--12：00)

(1). 基本概念：特征空间，决策树的划分，信息熵，信息增益, Gini指数

(2). 随机森林算法的原理.

(3). 练习：计算信息熵，Gini指数等.

### 4. 其他机器学习算法简介 (11月6日下午 14：00--17：00)

(1)k均值聚类，神经网络，SVM.

(2)模型评估，格点搜索等.

(3)练习：感知机分类器.

 

### 参考文献

[1]<http://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/>

[2]<https://www.jianshu.com/p/9ce5f3c3af99>