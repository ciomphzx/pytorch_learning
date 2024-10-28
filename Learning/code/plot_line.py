import matplotlib.pyplot as plt
import numpy as np

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['axes.grid'] = True

def set_figsize(figsize=(6, 4)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def has_one_axis(X):
    return hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__")

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, 
         xlim=None, ylim=None, xscale='linear', yscale='linear',  
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(6, 4), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def f(x):  
    return x ** 3 - 1 / x

x = np.arange(0.1, 3, 0.1)
plot(x, [f(x), 4 * x - 4], xlabel='x', ylabel='f(x)', legend=['f(x)', 'Tangent line (x=1)'])
plt.show()
