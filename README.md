# CS231n_Assignments
斯坦福大学公开课[cs231n](https://www.bilibili.com/video/BV1nJ411z7fe?from=search&seid=3617165519773000266).
## Assignment1 
实现k-Nearest Neighbor算法 SVM算法 Softmax_loss
## Assignment2
实现BathNormalization Dropout CNN 学习了pytorch的基本语法
## Assignment2
RNN的基本基本思想,实现LSTM.目标检测,风格迁移部分尚未完成
[toc]

## Assignment1

### KNN

训练的时候,KNN记住所有的训练数据,测试的时候,计算每个测试数据与训练集所有点的距离.然后按照距离从小到大的顺序,取出**k**个样本点,统计这K个元素中,哪个标签出现的最多,就可认为该测试数据属于这个类别.准确率不高,只有**27.4%**

顺带提了一下向量化计算的速度.

![image-20210903203341492](http://yirenwang.top:8873/images/2021/09/03/20210903203919.png)

### SVM

其实我没有掌握SVM的精髓,在作业里,SVM有点像是一个一层的神经网络,输入 X的维数是$N\times F$

权重矩阵W $F\times C$ 输出是 
$$
output = X W
$$


维度是$N\times C$也就是在各个分类上的得分,然后定义一个loss函数,设置学习率和正则,进行W的更新

准确率大概是 **40%**

![image-20210903210912569](http://yirenwang.top:8873/images/2021/09/03/20210903210915.png)

![image-20210903213118005](http://yirenwang.top:8873/images/2021/09/03/20210903213119.png)

这是SVM的权重可视化,对应各个类别

### Softmax classifier 

- 计算图

![image-20210904100002290](http://yirenwang.top:8873/images/2021/09/04/20210904100011.png)

- 防止溢出 

每一列都减去这一列的最大值.

![image-20210904100234929](http://yirenwang.top:8873/images/2021/09/04/20210904100236.png)

- 反向传播

softmax层的反向传播值是 $y_1 - t_1$ 具体步骤可看深度学习鱼书**P270**

![image-20210904104017750](http://yirenwang.top:8873/images/2021/09/04/20210904104018.png)

**这个只是书上的全部图，具体的实现细节读者可查阅附录，我觉得计算图比看公式理解要简单**

softmax classifier的准确率**33%**

## Assignment2

### BatchNormalization

![image-20210904105343489](http://yirenwang.top:8873/images/2021/09/04/20210904105344.png)

- 反向传播

![image-20210904150114094](http://yirenwang.top:8873/images/2021/09/04/20210904150117.png)

![image-20210904150140380](http://yirenwang.top:8873/images/2021/09/04/20210904150141.png)

反向传播时,梯度流是如何通过这个节点的,,我没看懂,留作日后交流学习

### dropout

这个比较简单,就是让某些神经元失活.前向反向传播都比较好实现.

### 不同优化器的

SGD的缺点

1. ![image-20210904155341618](http://yirenwang.top:8873/images/2021/09/04/20210904155342.png)

两个方向上，一个方向的梯度大，一个方向的梯度小，SGD在进行更新的时候，权重将在梯度大的方向上变化大，在梯度小的方向山变化小，然后出现“之”字运动

2. 无法应对鞍点

![image-20210904155552741](http://yirenwang.top:8873/images/2021/09/04/20210904155553.png)

<img src="C:\Users\Curiosity\Desktop\CS231n\Assignment123.assets\image-20210904160237083.png" alt="image-20210904160237083" style="zoom:50%;" />

Momentum是在原先的梯度方向上加上了一个速度，然后沿着这个相加的速度方向进行更新。

![image-20210904162036631](http://yirenwang.top:8873/images/2021/09/04/20210904162041.png)

![image-20210904162832128](http://yirenwang.top:8873/images/2021/09/04/20210904162833.png)

![image-20210904163152725](http://yirenwang.top:8873/images/2021/09/04/20210904163449.png)



