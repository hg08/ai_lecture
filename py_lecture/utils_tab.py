#coding: utf-8
def multip_tab():
	print "\n\n乘法表："
	for i in range(1,10):
		print
		for j in range(1,i+1):
			print "%d * %d = %d" %(i,j,i*j),

if __name__ == "__main__":
	multip_tab()