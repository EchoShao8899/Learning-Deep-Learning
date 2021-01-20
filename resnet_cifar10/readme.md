# ResNet for cifar10

## ResNet architecture

**Building block**

$H(x)=x+F(x)$. The building block tries to learn $F(x)$ instead of learning $H(x)$.

<img src="C:\research\learning\resnet_cifar10\image\1" alt="image-20210120145853258" style="zoom: 50%;" />

​                                          <img src="C:\research\learning\resnet_cifar10\image\2" alt="image-20210120150158115" style="zoom: 50%;" /> <img src="C:\research\learning\resnet_cifar10\image\3" alt="img" style="zoom:50%;" /> 

**Detail in implementation**

- $x$ and $F(x)$ must have the same size. $\rightarrow$  use the $1\times 1$ convolution layer to change $F(x)'s$ size
- embed the BN layer to speed up the training

**Network Design**

![image-20210120150725847](C:\research\learning\resnet_cifar10\image\4)

