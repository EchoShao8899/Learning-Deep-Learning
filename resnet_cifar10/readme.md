# ResNet for cifar10

## ResNet architecture

**Building block**

H(x)=x+F(x). The building block tries to learn F(x)  instead of learning  H(x) .
<img src=".\image\1.png" alt="1" style="zoom: 50%;" />

​                                 <img src=".\image\2.png" alt="2" style="zoom:50%;" /> <img src=".\image\3.png" alt="3" style="zoom:50%;" />

**Detail in implementation**

- x and F(x) must have the same size. Solution: use the 1*1 convolution layer to change F(x)'s size
- embed the BN layer to speed up the training

**Network Design**

<img src=".\image\4.png" alt="4" style="zoom: 67%;" />

