import torch

# 测试GPU是否可用
print(torch.cuda.is_available())

x = torch.arange(4.0, requires_grad=True)
print("x : ", x)
y = 2 * torch.dot(x, x)
print(y.size())
# 反向传播计算梯度，梯度存储在grad属性中
y.backward()
print("y = 2 * x^2, grad of x: ", x.grad)
# 梯度清零，梯度是累加的，不清零就会在原来梯度的基础上继续增加
x.grad.zero_()

y = x.sum()
print("x.sum(): ", y)
# 反向传播计算梯度，梯度存储在grad属性中
y.backward()
print("y = sum(x), grad of x: ", x.grad)

x.grad.zero_()
y = x * x
# 阻断梯度传播
u = y.detach()
z = u * x

z.backward(torch.ones_like(z))
print(u)
print(x.grad)

x.grad.zero_()
y.backward(torch.ones_like(y))
print(x.grad)