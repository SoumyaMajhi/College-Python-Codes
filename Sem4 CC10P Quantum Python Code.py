#.................Finite Square Well Potential................

"""
#..................BASIC CODE.........................
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import odeint,simps

#.........Transcendental Equations of Even Parity..........
feven1=lambda z: np.tan(z)
feven2=lambda z: np.sqrt((z0/z)**2.0-1.0)
feven =lambda z: np.tan(z)-np.sqrt((z0/z)**2.0-1)

#.........Transcendental Equations of Odd Parity..........
fodd1=lambda z: 1.0/np.tan(z)
fodd2=lambda z: -np.sqrt((z0/z)**2.0-1.0)
fodd =lambda z: 1.0/np.tan(z) + np.sqrt((z0/z)**2.0-1)

#....................Values of Constants..................
a=1.0          
V0=20.0           
m,hcut=1.0, 1.0 

z0 = (a/hcut)*np.sqrt(2.0*m*V0)
z = np.linspace(1e-4, z0, 1000, endpoint=False)

'''
From the general solution by imposing the boundary values of the SE in the
middle region we get a solution of Phi(x), they are of two types one is of
even and other of odd parity.
'''

#......................Transcendental Equations - 1 (Even)..............
plt.plot(z,feven1(z),label= 'tan(z)' )
plt.plot(z,feven2(z), label= r'$\sqrt{(\frac{z0}{z})^2-1}$' )
plt.ylim([-10,10])
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
plt.xticks(np.arange(0,z0,0.5))
plt.title('Solution of Transcendental Equation for Even Energy Eigen Values')
plt.grid()
plt.legend()
plt.show()

#evenRootGuess=eval(input("Enter guess roots(z) for even parity (comma seperated): "))
evenRootGuess=[1.3,4.0] #even guess roots from the graph

#......................Transcendental Equations - 2 (Odd).................
#plt.plot(z,fodd1(z),z,fodd2(z))
plt.plot(z,fodd1(z), label='cot(z)')
plt.plot(z,fodd2(z), label=r'$-\sqrt{(\frac{z0}{z})^2-1}$')
plt.ylim([-10,10])
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
plt.xticks(np.arange(0,z0,0.5))
plt.title('Solution of Transcendental Equation for Odd Energy Eigen Values')
plt.grid()
plt.legend()
plt.show()
#oddRootGuess=eval(input("Enter guess roots(z) for odd parity (comma seperated): "))
oddRootGuess=[2.2,5.2] #odd guess roots from the graph

#...................Using root function we find the actual roots.............
evenRoots= root(feven, evenRootGuess).x 
oddRoots = root(fodd, oddRootGuess).x    
zRoots=np.sort(np.concatenate((evenRoots,oddRoots)))

eigenEnergies= (zRoots * hcut/a)**2 /(2*m) - V0 

#print('The Eigen Values of Energies are :',eigenEnergies)
i=0
for i,e in enumerate(eigenEnergies):
    print ("Energy Level:",i,"Eigen Energy:",e)
    
#..............Definition of Potential Function.....................
def V(x):
    return 0.0 if abs(x)>a else -V0

x=np.linspace(-3*a,3*a,1000)

Vx=[V(i) for i in x]
plt.plot(x,Vx)
plt.xlabel('$x$')
plt.ylabel('$V(x)$')
plt.title('Finite Square Well Potential Plot')
plt.xlim(-3,3)
plt.ylim(-25,15)
plt.show()

#....................Definition of Schrodinger Equation..................
def SE(psipsidot,x):
    psi,psidot=psipsidot   
    psiddot=(2.0*m/hcut**2)*(V(x)-E)*psi 
    return [psidot,psiddot]

#.............Definition of Wave Function and Solution using odeint..........
def wavefunction(Energy):
    global E
    E=Energy
    psipsidot0=[0,1e-5]
    psi=odeint(SE,psipsidot0,x)[:,0]
    NC=1.0/np.sqrt(simps(psi**2,x))
    return x,NC*psi

'''
Plotting Bound State Wavefunctions Along With The Potential For Different
EigenEnergies With A Shift By Energy Value
'''
i=0
for energy in eigenEnergies:
    x,psi=wavefunction(energy)
    plt.plot(x,psi+energy, label='$E_{}$ = {:.2f}'.format(i,energy))
    i=i+1
    Vx=[V(i) for i in x]
plt.plot(x,Vx, label='$V(x)$')
plt.xlabel('$x$')
plt.ylabel('$\psi(x) , V(x)$')
plt.title('Bound State Wavefunctions & Potential For Different EigenEnergies')
plt.legend(loc='right')
plt.show()


#.....................ADDITIONAL PROBLEMS:.........................

#Comparing The Numerically Obtained Ground State Wavefunction With Analytical Result

#.......................Analytical...............................
energy=eigenEnergies[0]      #Ground State
l=np.sqrt(2*m*(energy+V0))/hcut
k=l*np.tan(l*a)
b=3*a

x1=np.linspace(-b,-a,50,endpoint=False)
x2=np.linspace(-a,a,50,endpoint=True)
x3=-x1[::-1]

x0=np.concatenate((x1,x2,x3))

psi2=np.cos(l*x2)
psi3=psi2[-1]*np.exp(-k*(x3-a))
psi1=psi3[::-1]

psi0=np.concatenate((psi1,psi2,psi3))

I=simps(psi0**2,x0)
psi0=1*psi0/np.sqrt(I)   #Normalization

plt.plot(x0,psi0,'*',label='Analytical')

#....................Numerical......................
x,psi4=wavefunction(energy)
plt.plot(x,psi4,label='Numerical')

plt.title('Analytical vs Numerical Result of Ground State $\psi(x)$')
plt.xlabel('$x$')
plt.ylabel('$\psi(x)$')
plt.legend()

plt.show()

psi_arr=[]
for energy in eigenEnergies:
    x,psi=wavefunction(energy)
    psi_arr.append(psi)

#..................Expectation Value of x: <x>..........
expectationValueOf_X=simps(psi_arr*x*psi_arr,x)
print('Expectation Values of x is <x>: ',expectationValueOf_X)

#............Expectation Value of x**2: <x^2>...................
expectationValueOf_X2=simps(psi_arr*x**2*psi_arr,x)
print('Expectation Values of x is <x^2>: ',expectationValueOf_X2)


#..............Check Orthonormality and print the matrix.........
print('Orthonormality matrix of the eigenstates: \n')
for i in range(len(eigenEnergies)):
	for j in range(len(eigenEnergies)):
		print('{:.4f}'.format(simps(psi_arr[i]*psi_arr[j],x)),end="	"),
	print('\n')

"""


#..............Harmonic Oscillator................

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import odeint, simps

#Simple Harmonic Oscillator

#-d2p/du2=(lambda-u^2)p
#constants
#left/right boundaries taken as b instead of infinite, b>>a
b=5
#position scale
x=np.linspace(-b,b,1000)

#definition of potential(Finite Potential well of depth V0)
V = lambda x: x**2

#Scrodinger Equation
def SE(psipsidot,x):
    psi,psidot=psipsidot
    psiddot=(V(x)-E)*psi
    return [psidot,psiddot]

#Solve the Scrondinger Equation(IVP) to find wavefunction
def wavefunction(Energy):
    global E
    E=Energy
    psi=odeint(SE,[0,1e-5],x)[:,0]
    return psi

#Shooting function to find psi at right boundary
def psiAtb(energy):
    return wavefunction(energy)[-1]

#Find zeros of a function v(u)
def findZeros(u,v):
    return u[np.where(np.diff(np.signbit(v)))[0]]

def norm(u,v):
    return v/np.sqrt(simps(v**2,u))

#PART1: plot bound state wavefunction for the eigenenergies along with potential

#search eigen states
E1,E2=0,25         #Searching initial and final boundaries to find eigen values
dE=0.1              #searching steps
energies=np.arange(E1,E2,dE) #Searching energy range to find eigen values

#plot shooting curve(optional)
psiAtbs=[psiAtb(i) for i in energies]
plt.plot(energies,psiAtbs,'+')  #show psi for of the energies in discrete points
plt.xlabel('$Energy$')
plt.ylabel('$\psi(b)$')
plt.xlim(0,3)
plt.title("Shooting curve for Harmonic Oscillator")
plt.grid()
plt.show()

#guess the eigen energes
rootGuesses=findZeros(energies,psiAtbs)
print('rootGuesses in Part1 are: ', rootGuesses)

#find eigenEnergies as exact roots of the shooting function
eigenEnergies=[root(psiAtb,i).x[0] for i in rootGuesses]
print('eigenEnergies in Part1 are: ', eigenEnergies)

n=0
#find and plot all possible eigen states solving SE at eigen energies
for i in eigenEnergies:
    plt.plot(x,norm(x,wavefunction(i))+i, label="$E_{}=${:.2f}".format({n},i))
    #plt.hlines(i,-b,b,linestyles='dashed')
    n=n+1
plt.plot(x,V(x),label="$V(x)$")
plt.legend(loc='right')
plt.xlabel('$x$')
plt.ylabel('$\psi(x) , V(x)$')
plt.title("Bound state wavefunction for eigenenergies along with potential")
plt.show()


#PART2: Plot 4 eigen state wavefunction with potential (and extra hlines)

#search 4 eigen states
c=4
E1=0       #Searching initial boundary to find eigen values
dE=0.1              #searching steps
energies=np.arange(E1,E2,dE) #Searching energy range to find eigen values

evs=[]
i=0
while i<c:
    E2=E1+dE
    if psiAtb(E1)*psiAtb(E2)<0:
        evs.append(root(psiAtb,E1).x[0])
        i=i+1
    E1=E2
print('eigenEnergies for 4 excited eigenstates in Part2: ', evs)
m=0
for i in evs:
    plt.plot(x,norm(x,wavefunction(i))+i,label='$E_{}=${:.2f}'.format({m},i))
    plt.hlines(i,-b,b,linestyles='dashed')
    m=m+1
plt.plot(x,V(x),label="$V(x)$")
plt.xlabel('$x$')
plt.ylabel('$\psi(x) , V(x)$')
plt.ylim(0,15) 
plt.xlim(-b,b)
plt.title("4 eigenstate wavefunction with potential")  
plt.legend()
plt.grid()
plt.show()