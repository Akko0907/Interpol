import matplotlib.pyplot as plt
import numpy as np
import Methods as interpol

def Problem1_1():
    # Use of LinSpline
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    x,y = interpol.Spline(xk,yk,1,step=0.02)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Linear Spline")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

def Problem1_2():
    # Use of cubic spline
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    x,y = interpol.Spline(xk,yk,3,step=0.02)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Cubic Spline")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

def Problem1_3():
    # Use of Lagrange
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    f = interpol.Lagrange(xk,yk)
    x = np.arange(1,3+0.02,0.02)
    y = f.at(x)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Lagrange Interpolation")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

def Problem1_4():
    # Use of Newton
    xk = np.linspace(1,3,11)
    yk = 1/xk +np.cos(3*xk)**2

    f = interpol.ForwardNewton(xk,yk)
    x = np.arange(1,3+0.02,0.02)
    y = f.at(x)

    x_real = np.linspace(1,3,100)
    y_real = 1/x_real +np.cos(3*x_real)**2

    plt.plot(x,y,c='purple',label="Forward Newton Interpolation")
    plt.scatter(xk,yk,c='k',label="Data points")
    plt.plot(x_real,y_real,c='r',label="Real function")
    plt.legend()
    plt.show()

