#.....................1st Order ODE...................

'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
def f(x,t): 
    dxdt = 5.0*x
    return dxdt
x0 =1.0
t=np.linspace(0,10,101)
s=odeint(f,x0,t)
print(s)
plt.plot(t,s)
plt.xlabel("Value of t")
plt.ylabel("Value of s (ode)")
plt.show()

'''

'''
OUTPUT
[[1.00000000e+00]
 [1.64872127e+00]
 [2.71828191e+00]
 ...
 [1.90734832e+21]
 [3.14468577e+21]
 [5.18471037e+21]]

'''



#........................2nd Order ODE (Classical Harmonic Oscillator - Damping) (21.02.2022).......................

'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

z=float(input('enter value of "z": '))           #damping factor
k=float(input('enter value of "k": '))           #spring factor
def f(u,t):
    x=u[0]
    y=u[1]
    dxdt=y
    dydt=-z*y-k*x
    return np.array([dxdt,dydt])
u0=[0,1]
t=np.linspace(0,500,10001)
s=odeint(f,u0,t)
print(s)

plt.plot(s[:,0],s[:,1])
plt.show()
plt.plot(t,s[:,0])
plt.show()

'''

'''
OUTPUT
enter value of "z": 0.02
enter value of "k": 0.49
[[ 0.          1.        ]
 [ 0.04996479  0.99838847]
 [ 0.09981849  0.99555626]
 ...
 [-0.00890059 -0.00249621]
 [-0.00901986 -0.00227428]
 [-0.00912797 -0.00204979]]

'''



#...................Coupled Differential Equation Using 3 Variables - Lorentz Attractor Model - Non-Linear Plot......................

'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def f(u,t):
    x=u[0]
    y=u[1]
    z=u[2]
    dxdt=10*(y-x)
    dydt=x*(28-z)-y
    dzdt=x*y-(8/3)*z
    return np.array([dxdt,dydt,dzdt])
uo=[1,0,0]

t=np.linspace(0,101,100001)
s=odeint(f,uo,t)
print(s)

#plt.plot(t,s[:,0],t,s[:,1],t,s[:,2])          #PLOT OF X(t),Y(t),Z(t)IN SAME GRAPH

plt.plot(s[:,0],s[:,2])                        #PLOT OF X(t)vs Z(t)
plt.xlabel("Value of X(t)")
plt.ylabel("Value of Z(t)")
plt.show()

'''

'''
OUTPUT
[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 9.90092652e-01  2.81247606e-02  1.41229057e-05]
 [ 9.80565956e-01  5.59464679e-02  5.58693144e-05]
 ...
 [-1.03134805e+01 -1.62501880e+01  2.01840013e+01]
 [-1.03734653e+01 -1.63147924e+01  2.02995863e+01]
 [-1.04334924e+01 -1.63785807e+01  2.04165197e+01]]

'''


#...............................GAUSSIAN FUNCTION.....................................

'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

x=np.linspace(-10,10,100)

N=float(input("Enter value of N: "))
sig=float(input("Enter value of sigma: "))
mu=float(input("Enter value of mu: "))

f=lambda x:(N/sig*np.sqrt(2*np.pi))*np.exp((-(x-mu)**2)/(2.0*sig**2))

#print(np.array([x,f(x)]))
plt.plot(x,f(x))
plt.show()

'''

#OUTPUT: Graph


#............................Area Under The Curve Of Gaussian Function............................

'''
import numpy as np
from scipy.integrate import quad,simps,trapz

import matplotlib.pyplot as plt

x=np.linspace(-2,2,200) 

N=float(input("Enter value of N: ")) #NORMALISATION CONSTANT
sig=float(input("Enter value of sigma: ")) #FWMH WIDTH
mu=float(input("Enter value of mu: "))#POSITION OF PEAK

f=lambda x: (N/sig*np.sqrt(2*np.pi))*np.exp((-(x-mu)**2)/(2.0*sig**2))

s1=quad(f,-np.inf,np.inf)
x1=np.linspace(-1,1,101)
s2=simps(f(x1),x1)
s3=trapz(f(x1),x1)
print('INTEGRATION BY QUAD: ',s1)
print('INTEGRATION BY SIMPSON: ',s2)
print('INTEGRATION BY TRAPZ: ',s3)

#print(np.array([x,f(x)]))

plt.plot(x,f(x))
plt.show()

'''


'''
OUTPUT
Enter value of N: 1
Enter value of sigma: 0.2
Enter value of mu: 0
INTEGRATION BY QUAD:  (6.283185307179587, 4.639297412243715e-08)
INTEGRATION BY SIMPSON:  6.283181703891748
INTEGRATION BY TRAPZ:  6.2831816274494745

'''

    

#................Numerically Verifying A Given Gaussian Integral..............

'''
import numpy as np
from scipy.integrate import quad,simps,trapz
import matplotlib.pyplot as plt

a=float(input("Enter value of a: ",)) 
b=float(input("Enter value of b: ",))
c=float(input("Enter value of c: ",))
x=np.linspace(-1,1,101)

l=lambda x: np.exp(-a*x**2+b*x+c)               #LHS of Gaussian Integration
r=lambda x: np.sqrt(np.pi/a)*np.exp((b**2/4*a)+c)  #RHS of Gaussian Integration
s1=quad(l,-np.inf,np.inf)
s2=simps(l(x),x)
s3=trapz(l(x),x)

print('VALUE OF LHS INTEGRATION BY QUAD: ',s1)
print('VALUE OF LHS INTEGRATION BY SIMPSON: ',s2)
print('VALUE OF LHS INTEGRATION BY TRAPZ: ',s3)
print('Value of RHS: ', r(x))

'''

'''
OUTPUT
Enter value of a: 1
Enter value of b: 1
Enter value of c: 1
VALUE OF LHS INTEGRATION BY QUAD:  (6.1864718159341905, 2.0214475437552896e-08)
VALUE OF LHS INTEGRATION BY SIMPSON:  4.598420051234154
VALUE OF LHS INTEGRATION BY TRAPZ:  4.598292642469942
Value of RHS:  6.186471815934188
'''


#....................Dynamical Integration Of Discrete Data (3rd March 2022)....................

'''
from scipy.integrate import simps,quad
import numpy as np
import matplotlib.pyplot as plt

x=[0.0,1.0,2.0,3.0,4.0] # x values
y=[0.0,1.0,4.0,9.0,16.0] # corrosponding y values

x1=[]
y1=[]
I=[]

for i in range(len(x)):
    x1.append(x[i])
    y1.append(y[i])
    print (x1,y1)
    I=simps(y1,x1)
    print(I)
plt.plot(x1,y1,label='Discrete data')  #ploting the Integration values performed manually

#now, original function definition instead of given points
r=np.linspace(0,5) 
plt.plot(r,r**2, label='Function')
plt.legend()

plt.show ()

'''

'''
OUTPUT
[0.0] [0.0]
0.0
[0.0, 1.0] [0.0, 1.0]
0.5
[0.0, 1.0, 2.0] [0.0, 1.0, 4.0]
2.6666666666666665
[0.0, 1.0, 2.0, 3.0] [0.0, 1.0, 4.0, 9.0]
9.166666666666666
[0.0, 1.0, 2.0, 3.0, 4.0] [0.0, 1.0, 4.0, 9.0, 16.0]
21.333333333333332

'''
#.............from Gaussian to delta function(by reducing the FWHM WIDTH)..............................

'''
import numpy as np
from scipy.integrate import quad,simps,trapz
import matplotlib.pyplot as plt

a=float(input("Give the value of the X axis range where the function is to be plotted: "))
x=np.linspace(-a,a,200)
N=float(input("Enter value of N: ")) #NORMALISATION CONSTANT
sig=float(input("Enter value of sigma(sig): ")) #FWHM WIDTH
mu=float(input("Enter value of mu: "))#POSITION OF PEAK

f=lambda x:(x**2+3.0*x)*(N/(sig*np.sqrt(2*np.pi)))*np.exp((-(x-mu)**2)/(2.0*sig**2))
#f=lambda x: (N/(sig*np.sqrt(2*np.pi)))*np.exp((-(x-mu)**2)/(2.0*sig**2))

s1=quad(f,-np.inf,np.inf)
print('INTEGRATION BY QUAD: ',s1)
plt.plot(x,f(x))
plt.show()
'''

'''
OUTPUT
Give the value of the X axis range where the function is to be plotted: 10
Enter value of N: 1
Enter value of sigma(sig): 0.2
Enter value of mu: 0
INTEGRATION BY QUAD:  (0.04000000000000008, 9.187619849545337e-09)
'''

#................Ploting of a Delta Function.....................

# import numpy as np
# from scipy.integrate import quad,simps,trapz
# import matplotlib.pyplot as plt

# eps=20     #epsilon
# a=0.01
# delta=lambda x : eps if abs(x)<(a/2) else 0
# delta=np.vectorize(delta) #it is used when we plot a piecewise function
# x=np.linspace(-2,+2,1000)

# plt.plot(x,delta(x))
# plt.show()


#..............The Legendre Polynomial Plot......................
'''
from scipy.special import legendre as l
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-1,1,1000)
for i in range(10):
    plt.plot(x,l(i)(x),label="$P_{}(x)$".format({i}))
plt.legend()
plt.show()
'''

#.....................Recursion Formula-1........................
'''
from scipy.special import legendre as l
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-1,1,1000)
n=int(input("Enter value of 'n': "))
LHS = (n+1)*l(n+1)(x)
RHS = (2*n+1)*x*l(n)(x)-n*l(n-1)(x)

plt.plot(x,LHS,'*',color="red",label="LHS")
plt.plot(x,RHS,'+',color="cyan",label="RHS")
plt.legend()
plt.show()
'''
'''
OUTPUT
Enter value of 'n': 4
'''

#........................Recursion Formula-2.........................

'''
from scipy.special import legendre as p
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-1,1,100)
n=5
LHS = lambda x : (1-x**2)*np.polyder(p(n))(x)
RHS = lambda x : (n+1)*x*p(n)(x)-(n+1)*p(n+1)(x)

plt.plot(x,LHS(x),'*',color="red",label="LHS")
plt.plot(x,RHS(x),'+',color="yellow",label="RHS")
plt.legend()
plt.show()
'''

#.......................Orthogonality.......................
'''
from scipy.special import legendre as P
import numpy as np
from scipy.integrate import simps as S

n=int(input("Give the value of n: "))
m=int(input("Give the value of m: "))
x=np.linspace(-1,1,1001)
y=(P(n)(x))*(P(m)(x))
I=S(y,x)
print(I)
'''
'''
OUTPUT-1
Give the value of n: 5
Give the value of m: 5
0.18181818364742836

OUTPUT-2
Give the value of n: 5
Give the value of m: 6
0.0
'''

#..................CONVOLUTION..................

'''
from scipy.special import legendre as p
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

x=np.linspace(-10,10,1000)

f = lambda x: np.exp((-(x-2)**2)/2)
g = lambda x: np.exp((-(x-1)**2)/2)

S = []
for i in range(len(x)):
    t = np.linspace(-x[i],x[i],1000)
    I= simps(f(x[i]-t)*g(x),t)
    S.append(I)
plt.plot(x,S)
plt.show()
'''

#...........convolution 2...............
'''
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
def f(x): return np.exp(-x)
def g(x): return np.sin(x)
x=np.linspace(0,20,101)
R=[]
for i in range(len(x)):
    t=np.linspace(-x[i],x[i],101)
    S=simps(f(x[i]-t)*g(t),t)
    R.append(S)
actual=(np.exp(-x)+np.sin(x)-np.cos(x))/2
plt.xlabel(r'$x$')
plt.ylabel(r'$R$')
plt.plot(x,R, label="convoluted")
plt.scatter(x,actual, label="actual")
plt.legend()
plt.show()
'''

#............bessel polynomial............

'''
n=np.arange(0,6,1)
import numpy as np 
from scipy.special import jv
import matplotlib.pyplot as plt 
x=np.linspace(-0,11.0,101)
for i in n:
    plt.plot(x,jv(i,x), label='$J_{}(x)$'.format({i}))
    plt.xlabel('$x$')
    plt.ylabel('$J_n(x)$')
    plt.legend()
plt.show()
'''

#............Bessel recursion relation 1...........

'''
n=int(input("enter value of n: "))
import numpy as np 
from scipy.special import jv,jvp
import matplotlib.pyplot as plt 
x=np.linspace(-21.0,21.0,101)
L=n*jv(n,x)+x*jvp(n,x)
R=x*jv(n-1,x)
plt.plot(x,L,label="L")
plt.plot(x,R,'*', label="R")
plt.xlabel('$x$')
plt.ylabel('$L & R$')
plt.legend()
plt.show()
'''

#...........Bessel recursion relation 2.......

'''
n=int(input("enter value of n: "))
import numpy as np 
from scipy.special import jv,jvp
import matplotlib.pyplot as plt 
x=np.linspace(-21.0,21.0,101)
L=(x**(-n))*jvp(n,x)-n*x**(-n-1)*jv(n,x)
R=-x**(-n)*jv(n+1,x)
plt.plot(x,L,label="L")
plt.plot(x,R,'*', label="R")
plt.xlabel('$x$')
plt.ylabel('$L & R$')
plt.legend()
plt.show()
'''


#...................Bessel orthogonality  complete.............

'''
import numpy as np
from scipy.optimize import root
from scipy.special import jv
import matplotlib.pyplot as plt
from scipy.integrate import quad
x=np.linspace (0,20,1001)
n=int(input('enter the value of n='))
def f(x): return jv(n,x)
plt.hlines(0,0,20)
plt.grid()
plt.plot(x,f(x))
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.show()
a=float(input('enter guessroot1: '))
b=float(input('enter guessroot2: '))
c=float(input('enter guessroot3: '))
S=root(f,np.array([a,b,c])).x
print("actual roots are: ", S)

a=float(input('enter value of a: '))
b=float(input('enter value of b: '))
L= lambda x: x*jv(n,a*x)*jv(n,b*x)
R= (jv(n+1,a)**2)/2
I=quad(L,0,1)[0]
print("LHS: ",I,"RHS: ",R if a==b else 0)
'''


#................FOURIER SERIES................. 

'''
import numpy as np
from scipy.integrate import simps 
import matplotlib.pyplot as plt 

def f(x): return x**2

n=1
xp=np.linspace(0,20,1001)
x=np.linspace(-np.pi, np.pi, 100)
a0=(1/np.pi)*simps(f(x),x)

def a(n): return (1/np.pi)*simps(f(x)*np.cos(x),x)
def b(n): return (1/np.pi)*simps(f(x)*np.sin(x),x)

s=0.5*a0

R=[]
for i in xp:
    for n in range(1,100):
        s=s+a(n)*np.cos(n*i)+b(n)*np.sin(n*i)
    R.append(s)
#print(R)
plt.plot(xp,R)
plt.xlabel('$x$')
plt.ylabel('$R$')
plt.show()
'''

#..................FOURIER SERIES OF SQUARE, SAWTOOTH, triangular  WAVES.............MONDAY 11.04.2022

'''
from scipy.signal import square,sawtooth,triang
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps 

L=100
f=5.0           #f=frequency, n=no. of terms
x=np.linspace(0,L,10000)

yarray=[square(2*np.pi*f*x/L), sawtooth(2*np.pi*f*x/L), triang(10000)]
i=1
labels=["Square", "Sawtooth", "Triangular"]
for y,j in zip(yarray,labels):
    a0=(2/L)*simps(y,x)
    def a(n): return (2/L)*simps(y*np.cos(2*np.pi*n*x/L),x)
    def b(n): return (2/L)*simps(y*np.sin(2*np.pi*n*x/L),x)
    s=0.5*a0
    s=s+sum([a(k)*np.cos(2*np.pi*k*x/L)+b(k)*np.sin(2*np.pi*k*x/L) for k in range(1,101)])
    plt.subplot(2,2,i)
    plt.plot(x,s)
    plt.plot(x,y)
    plt.title(j)
    i=i+1
plt.show()
'''

#....18.04.2022.....solving 1d heat eq with finite difference method.............

'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps 

x=np.linspace(0,1,101)
u=np.zeros(101)
u[50]=1
for i in range(100):        #Time Loop
    for j in range(1,100):
        u[j] += (u[j-1] - 2*u[j] + u[j+1])/4.0
plt.plot(x,u)
plt.xlabel('length ($x$)')
plt.ylabel('amount of heat($u$)')
plt.show()
'''

#............3d heat equation plot...............

'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps 
from matplotlib import cm 

x=np.linspace(0,1,101)
y=np.linspace(0,1,101)
u=np.zeros((101,101))
u[50,50]=1.0     #IC: Heating at middle

for t in range(1000):          #Time Loop
    for i in range(100):        
        for j in range(1,100):
            u[i,j] += (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])/4.0

X,Y=np.meshgrid(x,y)
plt.axes(projection="3d").plot_surface(X,Y,u, cmap=cm.jet, rstride=1, cstride=1)
plt.show()
'''