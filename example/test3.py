import torch 

# 1：创建张量的几种方法

# 创建一个空的张量，维度为2*3
v = torch.empty(2,3)

# 创建一个随机的张量，维度为2*3
v = torch.randn(2,3)

# 创建一个全为0的张量，维度为2*3
v = torch.zeros(2,3)

# 创建一个全为0的张量，维度为2*3，数据类型为long
v = torch.zeros(2,3,dtype=torch.long)
 
# cuda 表示使用GPU
#v = torch.randn(2,3).cuda()

# 2：张量的基本运算-加减乘除

x = torch.tensor([1,2,3] ,dtype=torch.float32)
y = torch.tensor([4,5,6] ,dtype=torch.float32)

print("加法：",x+y)
print("减法：",x-y)
print("乘法：",x*y) # 对应元素相乘 
print("除法：",x/y)

print("向量点积：",torch.dot(x,y)) # 1*4+2*5+3*6，作用是求向量的内积 

# 3：张量的基本运算-聚合操作

x = torch.tensor([[1,2,3],[4,5,6]] ,dtype=torch.float32) #(2,3)
print("张量x（平均数）：",x.mean()) # 所有元素的平均数 
print("张量x（最大值）：",x.max()) # 所有元素的最大值
print(x.mean(dim=0)) # 按列求平均值 按着第1维度求平均值（列）,按着第1维运算，其它维度不变 tensor([2.5000, 3.5000, 4.5000])
print(x.mean(dim=1)) # 按行求平均值 按着第2维度求平均值（行）,按着第1维运算，其它维度不变 tensor([2., 5.])
print(x.mean(dim=1,keepdim = True)) #  为了保持原有的维度 (2,1) tensor([[2.],[5.]])

# 4：张量的基本运算-cat 

x = torch.tensor([[1,2,3],[4,5,6]] ,dtype=torch.float32)  # (2,3)
y = torch.tensor([[7,8,9],[10,11,12]] ,dtype=torch.float32) #(2,3)

print( torch.cat([x,y],dim=0) ) # 按行拼接 n+1维发生变化，其它维度不变 结果维度为(4,3)
print( torch.cat([x,y],dim=1) ) # 按列拼接 n+1维发生变化，其它维度不变 结果维度为(2,6)

x = torch.zeros(2,1,3)
y = torch.zeros(2,2,3)
z = torch.zeros(2,3,3)
w = torch.cat([x,y,z],dim=1)  
print(w.shape) # (2,6,3)

# 5：自动微分 

# 自动计算一个函数关于一个变量在某一取值下的导数，仅仅需要执行tensor.backward（）函数，就可以通过反向传播算法（Back Propogation）自动完成

x = torch.tensor([2.0],requires_grad=True) # requires_grad=True 表示需要求导
y = torch.tensor([3.0],requires_grad=True)
z = (x+y )* (y-2)
print(z)   
z.backward() # 反向传播
print(x.grad ,y.grad) # x.grad 表示x的导数，y.grad 表示y的导数


x = torch.tensor( [[1.,0.],[-1.,1.]],requires_grad=True)  
z = x.pow(2).sum() # x.pow(2) 表示x的平方
z.backward() # 反向传播
print(x.grad) # x.grad 表示x的导数

# 6：调整张量形状

# (1) 进行view操作，张量必须是连续的，即需要满足is_contiguous() == True
x = torch.tensor([1,2,3,4,5,6])
print (x.shape) # (6,) 一维张量
x.view(2,3) # 调整张量形状为2*3 

# (2) 进行reshape操作，张量可以不连续，但是需要满足元素个数相同
# 定义一个不连续的张量
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

sliced_x = x[:, 1:]
print("是否连续:", sliced_x.is_contiguous())
print("张量形状:", sliced_x.shape)
print(sliced_x)
 
try:
    sliced_x.view(4) # 报错，因为x不是连续的，需要先调用contiguous()函数，将x转换为连续的张量
except Exception as e:
    print(e)
    print(sliced_x)
    sliced_x = sliced_x.reshape(4) # 调整张量形状为4 
    print("调整后：",sliced_x)

# （3）transpose 操作 只能对2维张量进行操作

x = torch.tensor([[1,2,3],[4,5,6]] ,dtype=torch.float32) # (2,3) 
x.transpose(0,1) # 转置操作，交换0维和1维的位置，结果为(3,2)

# （4）permute 操作 可以对多维张量进行操作
x = torch.tensor([[[1,2,3],[4,5,6]]] ,dtype=torch.float32) # (2,3) 
print(x.shape) #  (1,2,3)
print(x.permute(2,0,1))  

# 6：广播机制

# 矩阵相加，要求两个矩阵的形状相同
# 矩阵相乘，要求第一个矩阵的列数等于第二个矩阵的行数
# 如果形状不符合要求，可以通过广播机制进行调整，使得两个张量的形状相同，从而进行运算

x = torch.arange(1,4).view(3,1)
y = torch.arange(1,3).view(1,2)
print("广播：",x+y) #计算前，先将x和y的形状调整为相同的形状(3,2)，然后再进行计算  


# 7：张量的索引和切片

x = torch.arange(12).view(3,4)
print(x[1,3]) # 取出第2行第4列的元素
print(x[:,1])

# 8：张量的升维和降维


x = torch.tensor([1,2,3,4])
# 升维
y = torch.unsqueeze(x,dim=0) # 在第0维增加一个维度
print ("升维：",x.shape,y.shape,y)

# 降维
z = torch.squeeze(y,dim=0) # 去掉第dim维的维度
print ("降维：",y.shape,z.shape)

# 创建一个(3,4,1) 的三维张量，赋值为1
x = torch.ones(3,4,1)
print(x,x.shape)
z = torch.squeeze(x,dim=2) #只在维度为2的地方进行降维，且它的维度为1，非1的维度不会进行降维
print(z,z.shape)