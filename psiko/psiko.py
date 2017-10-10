# John Eargle
# 2017


import numpy as np


def square_comp(x, omega, k):
    """
    Single component of square function.
    """
    return (4.0/np.pi) * np.sin(2*np.pi*(2*k-1)*omega*x)/(2*k-1)

def square(x, omega, n):
    """
    Approximate a square wave with sin() series.
    """
    a = np.array([square_comp(x, omega, k) for k in range(1,n+1)])
    sum = np.zeros(len(x))
    for i in a:
        sum += i
    return sum

def square2(x, omega, n):
    """
    Approximate a square wave with sin() series.
    """
    a = np.array([square_comp(x, omega, k) for k in range(1,n+1)])
    return np.array([a[:n,i].sum() for i in range(len(x))])
