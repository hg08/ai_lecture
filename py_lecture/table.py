#!/usr/bin/env python
#coding:utf-8
# 9 * 9
def multip_tab():
    """1. 本函数可实现打印出一个乘法表这样的功能，成功实现了此功能的封装；
       2. 调用该函数的语法: 函数名()；
    """
    for i in range(1, 10):
        print
        for j in range(1, i+1):
            print "%d*%d=%d" % (i, j, i*j),

if __name__ == "__main__":
    print "这是一个乘法表："
    multip_tab()
