# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 16:01:35 2018

@author: lvsikai
"""
import os
import urllib
import threading

url_file='./photos.txt'
save_dir='../1photos'
f = open(url_file)
file_names = f.readlines()
f.close()

name_urls = [x.strip() for x in file_names]

class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, start1, name, threadID,end):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.start1 = start1
        self.end=end
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print "Starting " + self.name
        print_time(self.name, self.start1, self.end)
        print "Exiting " + self.name

def auto_down(url, filename):  
    try:  
        urllib.urlretrieve(url, filename)  

    except:  
        print 'Network conditions is not good.\nReloading.'  
        auto_down(url, filename) 



def print_time(threadName, start1, end):
    for name_url in name_urls:
        name_url=name_url.split(',')
        save_path=os.path.join(save_dir, name_url[0] + '.jpg')
        if int(name_url[0])<start1:
          continue
        if int(name_url[0])>end:
          break
        print int(name_url[0])
        print name_url[1]  
        abspath = os.path.abspath(save_path)  
        auto_down(name_url[1], abspath)

# 创建新线程
thread1 = myThread(330000, "Thread-1", 1,340000)
thread2 = myThread(340000, "Thread-2", 2,350000)
thread3 = myThread(350000, "Thread-3", 3,360000)
thread4 = myThread(360000, "Thread-4", 4,370000)
thread5 = myThread(370000, "Thread-5", 5,380000)
thread6 = myThread(380000, "Thread-6", 6,390000)
thread7 = myThread(390000, "Thread-7", 7,400000)
thread8 = myThread(400000, "Thread-8", 8,410000)
thread9 = myThread(410000, "Thread-9", 9,420000)
thread10 = myThread(420000, "Thread-10", 10,430000)
# 开启线程
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
thread8.start()
thread9.start()
thread10.start()
 
print "Exiting Main Thread"
