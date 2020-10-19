# MINST手写数字数据集预测

- 分别使用mlp和cnn模型完成了这项任务

- 网络训练代码请见mlp.py，cnn.py文件

- 测试代码请见test.py, test_for_CNN.py文件

- 可以使用我已经训练好的参数来跳过训练环节，参数分别储存在net_params.pkl和CNN_net_params.pkl文件中

- 实验结果

  ```python
  # mlp.py
  Epoch:  1 	Training Loss: 0.000754
  Epoch:  2 	Training Loss: 0.000618
  Epoch:  3 	Training Loss: 0.000541
  Epoch:  4 	Training Loss: 0.000529
  Epoch:  5 	Training Loss: 0.000524
  Epoch:  6 	Training Loss: 0.000522
  Epoch:  7 	Training Loss: 0.000521
  Epoch:  8 	Training Loss: 0.000521
  Epoch:  9 	Training Loss: 0.000521
  Epoch:  10 	Training Loss: 0.000520
  ```

  ```python
  # test.py
  Accuracy of the network on the test images: 85 %
  ```

  ```python
  # cnn.py
  Epoch:  1 	Training Loss: 0.000014
  Epoch:  2 	Training Loss: 0.000011
  Epoch:  3 	Training Loss: 0.000005
  Epoch:  4 	Training Loss: 0.000003
  Epoch:  5 	Training Loss: 0.000002
  ```

  ```python
  # test_for_CNN.py
  Accuracy of the network on the test images: 98 %
  ```

  

