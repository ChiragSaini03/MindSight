# importing the multiprocessing module 
import multiprocessing 
import time

def preprocess(num): 
	for i in range(num):
		if i==2:
			time.sleep(0.1)
		print(i,"t");

def speak(num): 
	for i in range(num):
		print(i,"f")

if __name__ == "__main__": 
	# creating processes 
	p1 = multiprocessing.Process(target=preprocess, args=(10, )) 
	p2 = multiprocessing.Process(target=speak, args=(10, )) 

	# starting process 1 
	p1.start() 
	# starting process 2 
	p2.start() 

	# wait until process 1 is finished 
	p1.join() 
	# wait until process 2 is finished 
	p2.join() 
 
	# both processes finished 
	print("Done!") 
