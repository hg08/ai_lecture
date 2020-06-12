# coding=utf-8
import ctypes
import pywinusb.hid as hid
import time
import chardet
##import sys
import os
import win32api
import win32con
import win32gui
import win32clipboard as w
import win32con
import clipboard

VID01=0x5a29
VID02=0x1EAB

ALL_DATA=[]

STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12


FOREGROUND_GREEN = 0x0a # green.
FOREGROUND_RED = 0x0c # red.
FOREGROUND_BLUE = 0x09 # blue.

#delay_time = 0.125
#delay_time = 0.0625
#delay_time = 0.03125
#delay_time = 0.015625
#delay_time = 0.0078125
delay_time = 0.00390625
# get handle
# about cmd output 
std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

def set_cmd_text_color(color, handle=std_out_handle):
    Bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    return Bool

#reset white
def resetColor():
    set_cmd_text_color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)



#红色 
def printRed(mess):
    set_cmd_text_color(FOREGROUND_RED)
    print mess
    resetColor()


#绿色
#green
def printGreen(mess):
    set_cmd_text_color(FOREGROUND_GREEN)
    print mess
    resetColor()

#白色
#white
def printWhite(mess):
    set_cmd_text_color(FOREGROUND_WHITE)
    print mess
    resetColor()

    
# Get Text from clipboard
def getText():
    w.OpenClipboard()
    d = w.GetClipboardData(win32con.CF_TEXT)
    w.CloseClipboard()
    return d


# Set Text for clipboard
def setText(aString):
    w.OpenClipboard()
    w.EmptyClipboard()
    w.SetClipboardData(win32con.CF_TEXT, aString)
    w.CloseClipboard()

# press Ctrl+v
def press_Ctrl_v():
    win32api.keybd_event(17,0,0,0)  #ctrl键位码是17
    win32api.keybd_event(86,0,0,0)  #v键位码是86
    win32api.keybd_event(86,0,win32con.KEYEVENTF_KEYUP,0) #释放按键
    win32api.keybd_event(17,0,win32con.KEYEVENTF_KEYUP,0)

# press tab
def press_tab():
    win32api.keybd_event(9,0,0,0)  #tab键位码是9
    win32api.keybd_event(9,0,win32con.KEYEVENTF_KEYUP,0) #释放按键


def delay():
    global delay_time
    time.sleep(delay_time)

def ask_delay_time():
    global delay_time
    print "Input delay time:"
    delay_time=float(raw_input())

def sample_handler(data):   
    global ALL_DATA
    # Data transfer is over 
    if(data[-1]==0):
        ALL_DATA=ALL_DATA+data[2:-6]
        #print "All data:"
        #print ALL_DATA
        # Use data to form a string
        s="".join(map(chr,ALL_DATA))
        # Clear ALL_DATA
        ALL_DATA=[]
        # Get the encoding of the string
        En_Coding=chardet.detect(s)
        En_Coding=En_Coding['encoding']
        #print En_Coding
        if(En_Coding=='ascii'):
                    #Just insert
                    clipboard.copy(s)
                    press_Ctrl_v()
                    time.sleep(0.03125)
                    # clear clipboard when work done
                    clipboard.copy('')
        else:

                    #print s
                    if ("AALIPAY" in s)and("→" in s):
                        # Fill tables
                        a=s.split('→')  # DO NOT CHANGE THIS LINE if you do not know 
                            # Deleta extra char
                        str1=a[0][5:]
                        # Add try-except to haddle error 
                        try:
                            temp1=str1.decode(En_Coding)
                            section1=temp1.encode('gb2312')
                            str2=a[1][1:]
                            temp2=str2.decode(En_Coding)
                            section2=temp2.encode('gb2312')
                            str3=a[2][1:]
                            temp3=str3.decode(En_Coding)
                            section3=temp3.encode('gb2312')
                            str4=a[3][1:]
                            temp4=str4.decode(En_Coding)
                            section4=temp4.encode('gb2312')
                            #print section1
                            clipboard.copy(section1)
                            press_Ctrl_v()
                            delay()
                            press_tab()
                            #print section2
                            clipboard.copy(section2)
                            press_Ctrl_v()
                            delay()
                            press_tab()
                            #print section3
                            clipboard.copy(section3)
                            press_Ctrl_v()
                            delay()
                            press_tab()
                            #print section4
                            clipboard.copy(section4)
                            press_Ctrl_v()
                            delay()
                            # clear clipboard when work done
                            clipboard.copy('')
                        except:
                            #print "Decode Error"  
                            pass
                    else:
                        # Other utf8 code just covert code and past
                        s=s.decode(En_Coding)
                        s=s.encode('gb2312')
                        clipboard.copy(s)
                        press_Ctrl_v()
                        time.sleep(0.03125)
                        # clear clipboard when work done
                        clipboard.copy('')

    # Data transfer not completed : data[-1]==1
    else:
        # Remove useless data and connect data
        ALL_DATA=ALL_DATA+data[2:-6]
        







def welcome():
    print  "********************************"
    print u"*** 上海鹰捷智能科技有限公司 ***"
    print u"***      发票录入助手        ***"
    print  "********************************"


def test_performance():
    t0 = time.clock()
    ans = 123456789012345678901**100456
    t1 = time.clock()
    return t1-t0

def ask_user_input():
    global delay_time
    print u"输入软件反应级别编号,以回车键结束"
    print u"[1]:快速"
    print u"[2]:较快"
    print u"[3]:常速"
    print u"[4]:较慢"
    print u"[5]:慢速"
    user_input = raw_input()
    if user_input == '1':
        delay_time = 0.015625
    elif user_input == '2':
        delay_time = 0.03125
    elif user_input == '3':
        delay_time = 0.25
    elif user_input == '4':
        delay_time = 0.5
    elif user_input == '5':
        delay_time = 1
    else:
        print u"请输入正确的级别编号:"
        ask_user_input()
        
while True:
    device01=hid.HidDeviceFilter (vendor_id=VID01)
    device01=device01.get_devices()
    if len(device01):
        os.system("cls")
        welcome()
        #print test_performance()
        #ask_delay_time()
        ask_user_input()
        printGreen(u"已连接，欢迎使用！")
        device01=device01[0]
        device01.open()
        device01.set_raw_data_handler(sample_handler)
        while device01.is_plugged():
            time.sleep(2)
    else:
        device02=hid.HidDeviceFilter(vendor_id=VID02)
        device02=device02.get_devices()
        if len(device02):
            os.system("cls")
            welcome()
            ask_delay_time()
            printGreen(u"已连接，欢迎使用！")
            device02=device02[0]
            device02.open()
            device02.set_raw_data_handler(sample_handler)
            while device02.is_plugged():
                time.sleep(2)
        else:
            os.system("cls")
            welcome()
            printRed(u"设备已断开，请连接。")
            time.sleep(2)


