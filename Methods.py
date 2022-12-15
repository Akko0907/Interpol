import numpy as np


#=======================================================================================
#=======================================================================================


import sys
sys.path.append('C:\\Users\\gubis\\projects\\MAP_codes')
from Linear.Methods import GaussE
def DirectPol(x: np.ndarray, y: np.ndarray, round: int=3) -> np.ndarray:
    n = len(x)-1
    A = np.stack([x for i in range(n+1)],axis=1)
    A = np.power(A,[i for i in range(n+1)])
    
    alphas = GaussE(A,y)
    
    return np.round(alphas,round)


#=======================================================================================
#=======================================================================================


class Lagrange():
    def __init__(self: object, data_x: np.ndarray, 
                 data_y: np.ndarray, order: int=None):
        self.__x = data_x
        self.__y = data_y
        self.N = len(data_x)

        if type(order) is not int and order is not None:
            raise TypeError(f"`order` must be an integer, but is: {type(order)}")
        if type(order) is int:
            if order>=self.N:
                raise ValueError(f"`order` must be less than the length of data: {self.N}")
        
        self.__order = self.N-1 if order==None else order
        if self.__order<=20:
            self.__div = self.Prod_dx()
            self.excep = False
        else:
            self.excep = True


    def __str__(self) -> str:
        return f"Lagrange Interpolation constructed with:\nx={self.__x}\ny={self.__y}"
    def __repr__(self) -> str:
        return f"Lagrange Interpolation function"


    @property
    def x_data(self):
        return self.__x
    @property
    def y_data(self):
        return self.__y
    @property
    def order(self):
        return self.__order


    @order.setter
    def order(self,ord: int=None):
        if type(ord) is not int and ord is not None:
            raise TypeError(f"`order` must be an integer or None, but is: {type(ord)}")
        if type(ord) is int:
            if ord>=self.N:
                raise ValueError(f"`order` must be less than the length of data: {self.N}")

        self.__order = self.N-1 if ord==None else ord
        self.__div = self.Prod_dx()
        self.excep = False


    def Prod_dx(self, x: float=None):
        
        prod = np.zeros(self.__order+1)

        if x==None:
            for i in range(self.__order+1):
                term = self.__x[i] - self.__x[self.__x!=self.__x[i]][:self.__order]
                prod[i] = np.prod(term)
        else:
            for i in range(self.__order+1):
                term = x - self.__x[self.__x!=self.__x[i]][:self.__order]
                prod[i] = np.prod(term)

        return prod


    def at(self,x: np.ndarray) -> np.ndarray:
        if self.excep:
            try:
                raise Exception
            except:
                print("\nWe do not recommend interpolation with order>20, if you still want to try it,")
                print("please, set the order manually and try again. Else try the Spline() function\n")
        else:
            f = lambda x: np.dot( self.__y[:self.__order+1], self.Prod_dx(x)/self.__div )
        
            try:
                iter(x)
            except TypeError:  
                L = f(x)
            else:
                L = np.array([f(value) for value in x])

            return L


#=======================================================================================
#=======================================================================================


class ForwardNewton():
    def __init__(self: object, data_x: np.ndarray, 
                 data_y: np.ndarray, order: int=None):
        self.__x = data_x
        self.__y = data_y
        self.N = len(data_x)

        if type(order) is not int and order is not None:
            raise TypeError(f"`order` must be an integer, but is: {type(order)}")
        if type(order) is int:
            if order>=self.N:
                raise ValueError(f"`order` must be less than the length of data: {self.N}")
        
        self.__order = self.N-1 if order==None else order
        self.__dfs = self.DivDiff()


    def __str__(self) -> str:
        return f"Newton Interpolation constructed with:\nx={self.__x}\ny={self.__y}"
    def __repr__(self) -> str:
        return f"Newton Interpolation function"


    @property
    def get_x(self):
        return self.__x
    @property
    def get_y(self):
        return self.__y
    @property
    def order(self):
        return self.__order


    @order.setter
    def order(self,ord: int=None):
        if type(ord) is not int and ord is not None:
            raise TypeError(f"`n` must be an integer or None, but is: {type(ord)}")
        if type(ord) is int:
            if ord>=self.N:
                raise ValueError(f"`n` must be less than the length of data: {self.N}")

        self.__order = self.N-1 if ord==None else ord
        self.__dfs = self.DivDiff()


    def DivDiff(self):
        # Generates divided differences table
        dfs = np.zeros((self.__order+1,self.__order+1))
        dfs[0] = self.__y[:self.__order+1]
        for i in range(1,self.__order+1):
            for j in range(self.__order-i+1):
                dfs[i][j] = (dfs[i-1][j] - dfs[i-1][j+1]) / (self.__x[j] - self.__x[i + j])
        return dfs


    def at(self,x: np.ndarray) -> np.ndarray:

        g = lambda u: np.array([ np.prod(u-self.__x[:i]) for i in range(1,self.__order+1) ]) 
        f = lambda t: self.__y[0] + np.dot(t,self.__dfs[1:,0])

        try:
            iter(x)
        except TypeError:  
            N = f(g(x))
        else:
            N = np.array([f(g(u)) for u in x])

        return N


#=======================================================================================
#=======================================================================================


class Stirling():
    def __init__(self: object, data_x: np.ndarray, 
                 data_y: np.ndarray):
        self.__x = data_x
        self.__y = data_y
        self.n = len(data_x)
        self.__dfs_center = self.DivDiff()
        self.h = data_x[1]-data_x[0]


    def __str__(self) -> str:
        return f"Stirling Interpolation constructed with:\nx={self.__x}\ny={self.__y}"
    def __repr__(self) -> str:
        return f"Stirling Interpolation function"


    @property
    def get_x(self):
        return self.__x    
    @property
    def get_y(self):
        return self.__y


    def DivDiff(self):
        # Generates divided differences table
        dfs = np.zeros((self.n,self.n))
        for i in range(self.n):
            dfs[:,i] = np.append( np.diff(self.__y,i), np.zeros(i) )

        df_center = np.zeros(self.n)
        for j in range(self.n):
            if j%2==0:
                i = (self.n-1-j)//2
                df_center[j] = dfs[i][j]
            else:
                k = dfs[:,j]
                k = k[k!=0]
                i = len(k)//2
                df_center[j] = (dfs[i][j]+dfs[i-1][j])/2

        return df_center
    
    def Terms(self, x):             
        s = ( x - self.__x[(self.n-1)//2] ) / self.h        

        prods = np.ones(self.n)*s 
        prods[0] = 1
        for i in range(1,self.n):
            if i%2==0:
                prods[i] = prods[i-1] * s/i
            else:
                prods[i] = prods[i-1] * (s**2-(i//2)**2) / (i*s) 
        return prods


    def at(self, x: np.ndarray) -> np.ndarray:
        try:
            iter(x)
        except TypeError:  
            S = np.prod(self.Terms(x),self.__dfs_center)
        else:
            S = np.array([np.prod(self.Terms(u),self.__dfs_center) for u in x])

        return S


#=======================================================================================
#=======================================================================================


def Spline(x_data,y_data,order,step=0.01) -> tuple:
    
    N = len(x_data)
    x = np.array([])
    y = np.array([])
    
    for i in range(N-order):
        
        f = Lagrange(x_data[i:i+order+1],y_data[i:i+order+1])
        
        if i==0:
            x_chunk = np.arange(x_data[i],x_data[i+order]+step,step)
        elif i<N-3:
            x_chunk = np.arange(x[-1]+step,x_data[i+order]+step,step)
        else:
            x_chunk = np.arange(x[-1]+step,x_data[i+order]+step,step)
            
        y_chunk = f.at(x_chunk)
        x = np.concatenate([x,x_chunk],axis=None)
        y = np.concatenate([y,y_chunk],axis=None)
    return (x,y)