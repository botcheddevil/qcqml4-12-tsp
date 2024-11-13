from numpy import ndarray


const = 1
op = "+"
approxRatios = {}

def setConst(newConst):
    global const
    const=newConst

def incConst():
    global const
    const+=1

def setOp(opNew):
    global op
    op = opNew

def calculate(distances: ndarray):
    global op
    global const
    return (distances.max() + const) if op == "+" else distances.max() * const

def key():
    return f"MD{op}{const}"