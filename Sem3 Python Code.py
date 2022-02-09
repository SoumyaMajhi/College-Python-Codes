#....................Analytical Method.................#
# def f(x): return x**2-96.0# defining a function.
# x=0.0
# t=0.0001
# while f(x)<0.0001:# it will run untill it crosses the 0.0001
# 	x=x+t
# 	#print(x, f(x))
# print(x, f(x))

# OUTPUT
# 9.797999999990504 0.0008039998139111049



#.....................Bisection Method...................#
# import sys, os
# def f(x): return x**2-4.0
# a=eval(input("enter lower limit: "))
# b=eval(input("enter upper limit: "))
# if f(a)*f(b)>0:
#     print("No root is available within this range")
#     sys.exit()
# while abs(a-b)>=0.001:
#     xm=(a+b)*0.5
#     if f(xm)==0:
#         print("Root = ",xm)
#         sys.exit()
#     if f(a)*f(xm)<0:
#         b=xm                              #Left Half
#     else:
#         a=xm                              #Right Half
# print("Root = ", (a+b)*0.5)

# OUTPUT
# enter lower limit: 1
# enter upper limit: 5
# Root =  2.0



#..................Newton-Raphson Method..................#
# import sys
# def f(x):return x**2-4.0                             #function
# def h(x):return 2*x                                  #derivative
# x=float(input("enter the value of approximate root: "))
# if f(x)==0:
# 	print("Root = ", x)
# 	sys.exit()
# while f(x)>0.0001:
# 	x=x-f(x)/h(x)
# print("Root = ", x)

# OUTPUT
# enter the value of approximate root: 5
# Root =  2.0000051812194735



							#INTERPOLATION


#...................Newton's Forward Interpolation...................#
# x=[5.0,10.0,15.0,20.0,25.0,30.0]
# y=[45.0,105.0,174.0,259.0,364.0,496.0]
# d=[]
# xn = float(input("enter a value of x: "))
# t=(xn-x[0])/5.0            #here, put required value of x instead of 18.0
# sum=y[0]
# coef=t
# k=1.0
# for i in range(len(y),1,-1):
# 	for j in range(i-1):
# 		dif=y[j+1]-y[j]
# 		d.append(dif)
# 	sum=sum+coef*d[0]
# 	coef=(coef*(t-k))/(k+1)							#updating the coef
# 	k=k+1
# 	y=d
# 	d=[]
# print("Interpolated Value = ", sum)

# OUTPUT
# Interpolated Value =  222.826688


#..................Newton's Backward Interpolation.................#
# x=[5.0,10.0,15.0,20.0,25.0,30.0]
# y=[45.0,105.0,174.0,259.0,364.0,496.0]
# d=[]
# xn = float(input("enter a value of x: "))
# n=len(x)-1
# t=(xn-x[n])/5.0
# sum=y[n]
# coef=t
# k=1.0
# for i in range (len(y),1,-1):
# 	for j in range(i-1):
# 		dif=y[j+1]-y[j]
# 		d.append(dif)
# 	sum=sum+coef*d[j]
# 	coef=(coef*(t+k))/(k+1)
# 	k=k+1
# 	y=d
# 	d=[]
# print("Interpolated Value = ", sum)



							#INTEGRATION

#.......................Rectangular Method..........................#

# def f(x): return x**2
# b=float(input('enter upper limit: '))
# a=float(input('enter lower limit: '))
# n=float(input('enter number of sub-intervals: '))
# h = (b-a)/n
# sum=0
# y=[]
# while (a<=i<=b):
# 	y=f(i)
# 	for j in range(n+1):
# 		sum=h*y


	
# print (sum)

# def f(x): return 3*x


# def rect(f,a,b,r):
#     carea=0

#     a=float(a)
#     b=float(b)
#     r=float(r)

#     i=(b-a)/r

#     tx=a
#     lx=a+i

#     while (a<=lx<=b) or (a>=lx>=b):
#         area=f((tx+lx)/2)*i
#         carea+=area

#         lx+=i
#         tx+=i

#     return carea
# print()



#...................Trapezoidal Rule.................#

# def f(x): return x
# b=float(input('enter upper limit: '))
# a=float(input('enter lower limit: '))
# h=b-a
# sum= 0.5*(f(a)+f(b))*h
# print("Value of Integral = ", sum)

# OUTPUT
# enter upper limit: 6
# enter lower limit: 2
# Value of Integral = 16.0

					#Composite

# def f(x): return x**2
# b=float(input('enter upper limit: '))
# a=float(input('enter lower limit: '))
# n=int(input('enter no of division: '))
# h=(b-a)/n
# sum=(f(b)+f(a))*0.5
# for i in range (1,n):
# 	x=a+i*h
# 	sum=sum+f(x)
# print("Value of Integral = ", h*sum)

# OUTPUT
# enter upper limit: 10
# enter lower limit: 2
# enter no of division: 100
# Value of Integral =  330.67519999999996


#...................Simpson's 1/3 Rule..................#
# def f(x): return x**2
# b=float(input('enter upper limit: '))
# a=float(input('enter lower limit: '))
# h=(b-a)/2
# y0=f(a)
# y1=f(0.5*(a+b))
# y2=f(b)
# sum= (h/3.0)*(y0+4*y1+y2)
# print("Value of Integral =  ", sum)

# OUTPUT
# enter upper limit: 10
# enter lower limit: 2
# Value of Integral = 330.66666666666663


						#Composite

# def f(x): return x**2
# b=float(input('enter upper limit: '))
# a=float(input('enter lower limit: '))
# n=int(input('enter number of division: '))
# h=(b-a)/n
# sum1=f(a)+f(b)
# sum2=0.0
# for i in range (1,n,2):
# 	x=a+i*h
# 	sum2=sum2+f(x)
# sum3=0.0
# for j in range (2,n,2):
# 	x=a+j*h
# 	sum3=sum3+f(x)
# I=(h/3.0)*(sum1+(4*sum2)+(2*sum3))
# print("Value of Integral = ", I)

# OUTPUT
# enter upper limit: 10
# enter lower limit: 2
# enter number of division: 10
# Value of Integral =  330.6666666666667


#..................Integration by SciPy Module.............#
# import numpy as np
# from scipy import integrate
# x=np.arange(2,11)
# y = x**2
# I=integrate.simps(y,x)
# print("Value of Integral =  ", I)

# OUTPUT
#Value of Integral =   330.66666666666663


					#ODE

#.................Euler's Method..................#
# def f(x,y):return 3*x*y       			#Defining the function
# x, y = 0.0, 1.0    						#Initial values
# h=0.01
# X, Y = [x], [y]   								#Step size
# for i in range(201):
# 	y=y+h*f(x,y)						#Updating by Euler's formula
# 	x=x+h
# print(x, y)

# OUTPUT
# 2.010000000000001 369.6691477543691


#...............RK2...............#
# def f(x,y):return 3*x*y
# x,y = 0.0, 1.0
# h = 0.01
# for i in range(201):
# 	k1=h*f(x,y)
# 	k2=h*f(x+h,y+k1)
# 	y=y+0.5*(k1+k2)
# 	x=x+h
# print(x,y)

# OUTPUT
# 2.010000000000001 427.68060123959447


#..............RK4...................#
# def f(x,y): return 3*x*y
# x,y = 0.0, 1.0
# h = 0.01
# for i in range(201):
# 	k1=h*f(x,y)
# 	k2=h*f(x+h*0.5,y+k1*0.5)
# 	k3=h*f(x+h*0.5,y+k2*0.5)
# 	k4=h*f(x+h,y+k3)
# 	y=y+(k1+2*k2+2*k3+k4)/6.0
# 	x=x+h
# print(x, y)

# OUTPUT
# 2.010000000000001 428.4396065649443


#......................COMPARISON......................#
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# d=np.linspace(0,2.01,201)
# d3=np.exp(1.5*d**2)
# d1=[]
# d2=[]
# d4=[]
# #Euler Method
# def f(x,y): return 3*x2*y2
# x2=0.0
# y2=1.0
# h=0.01
# for i in range (201):
# 	y2=y2+h*f(x2,y2)
# 	x2=x2+h
# 	d4.append(y2)
# print (y2)
# #Rk 2 Method 
# def f(x,y): return 3*x*y
# x=0.0
# y=1.0
# h=0.01
# for i in range (201):
# 	k1=h*f(x,y)
# 	k2=h*f(x+h,y+k1)
# 	y=y+0.5*(k1+k2)
# 	x=x+h
# 	d1.append(y)
# print (y)
# #RK-4 Method
# def f(x,y): return 3*x1*y1
# x1=0.0
# y1=1.0
# h=0.01
# for i in range (201):
# 	k11=h*f(x1,y1)
# 	k22=h*f(x1+h/2.0,y1+k11/2.0)
# 	k3=h*f(x1+h/2.0,y1+k22/2.0)
# 	k4=h*f(x1+h,y1+k3)
# 	y1=y1+(1/6.0)*(k11+2*k22+2*k3+k4)
# 	x1=x1+h
# 	d2.append(y1)
# print (y1)
# plt.plot(d,d1,'*',label='RK2')
# plt.plot(d,d2,'.',label='RK4')
# plt.plot(d,d3,label='EXACT') 
# plt.plot(d,d4,'+',label='EULER')
# plt. legend(fontsize=10) 
# plt.show()