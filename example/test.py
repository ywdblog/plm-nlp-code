import torch 
x = torch.tensor([[1,2,3],[4,5,6]] ,dtype=torch.float32) #(2,3)
print("张量x（平均数）：",x.mean()) # 所有元素的平均数 
print("张量x（最大值）：",x.max()) # 所有元素的最大值
print(x.mean(dim=0)) # 按列求平均值 按着第1维度求平均值（列）,按着第1维运算，其它维度不变 tensor([2.5000, 3.5000, 4.5000])
print(x.mean(dim=1)) # 按行求平均值 按着第2维度求平均值（行）,按着第1维运算，其它维度不变 tensor([2., 5.])
print(x.mean(dim=1,keepdim = True)) #  为了保持原有的维度 (2,1) tensor([[2.],[5.]])