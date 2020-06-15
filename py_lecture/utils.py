#!/usr/bin/env python
#coding: utf-8

#学习目标
#1. 学会导入模块
#2. 学会调用模块中的函数

#为了能永久地保存程序，需要将代码写入文件中。这样的文件通常称为模块。
# 模块是一个包含了Python语句的简单文本文件。

#如函数是定义在模块中，必须先导入这个函数，再调用它。导入函数的语法为
#    from 模块名 import 函数名


def times(x,y):
    result = x * y
    return result

#9*9
def multip_tab():
    """1. 本函数可实现打印出一个乘法表这样的功能，成功实现了此功能的封装；
       2. 调用该函数的语法: 函数名()；
    """
    for i in range(1, 10):
        print
        for j in range(1, i+1):
            print "%d*%d=%d" % (i, j, i*j),
