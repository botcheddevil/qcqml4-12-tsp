from numpy import ndarray


const = 1
op = "+"
approxRatios = {}

def reset():
    global const
    const=1

def incConst():
    global const
    const+=1

def toggleOp():
    global op
    op = "x" if op=="+" else "x"

def calculate(distances: ndarray):
    global op
    global const
    return (distances.max() + const) if op == "+" else distances.max() * const

def key():
    return f"MD{op}{const}"