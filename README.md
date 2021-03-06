# MTCNN 
MTCNN方法可以概括为：图像金字塔+3阶段级联CNN，如下图所示  
![output](https://github.com/WwLuo-1024/MTCNN/blob/master/FYPMuQ.png)  
对输入图像建立金字塔是为了检测不同尺度的人脸，通过级联CNN完成对人脸 由粗到细（coarse-to-fine） 的检测，所谓级联指的是 前者的输出是后者的输入，前者往往先使用少量信息做个大致的判断，快速将不是人脸的区域剔除，剩下可能包含人脸的区域交给后面更复杂的网络，利用更多信息进一步筛选，这种由粗到细的方式在保证召回率的情况下可以大大提高筛选效率。下面为MTCNN中级联的3个网络（P-Net、R-Net、O-Net），可以看到它们的网络层数逐渐加深，输入图像的尺寸（感受野）在逐渐变大12→24→48，最终输出的特征维数也在增加32→128→256，意味着利用的信息越来越多。  
![output](https://github.com/WwLuo-1024/MTCNN/blob/master/FYPs4x.png)  
工作流程是怎样的？
首先，对原图通过双线性插值构建图像金字塔，可以参看前面的博文《人脸检测中，如何构建输入图像金字塔》。构建好金字塔后，将金字塔中的图像逐个输入给P-Net。    
  ·P-Net：其实是个全卷积神经网络（FCN），前向传播得到的特征图在每个位置是个32维的特征向量，用于判断每个位置处约\(12\times12\)大小的区域内是否包含人脸，如果包含人脸，则回归出人脸的Bounding Box，进一步获得Bounding Box对应到原图中的区域，通过NMS保留分数最高的Bounding box以及移除重叠区域过大的Bounding Box。  
  ·R-Net：是单纯的卷积神经网络（CNN），先将P-Net认为可能包含人脸的Bounding Box 双线性插值到\(24\times24\)，输入给R-Net，判断是否包含人脸，如果包含人脸，也回归出Bounding Box，同样经过NMS过滤。  
O-Net：也是纯粹的卷积神经网络（CNN），将R-Net认为可能包含人脸的Bounding Box 双线性插值到\(48\times 48\)，输入给O-Net，进行人脸检测和关键点提取。
  
需要注意的是：
  
1.face classification判断是不是人脸使用的是softmax，因此输出是2维的，一个代表是人脸，一个代表不是人脸  
2.bounding box regression回归出的是bounding box左上角和右下角的偏移\(dx1, dy1, dx2, dy2\)，因此是4维的  
3.facial landmark localization回归出的是左眼、右眼、鼻子、左嘴角、右嘴角共5个点的位置，因此是10维的  
4.在训练阶段，3个网络都会将关键点位置作为监督信号来引导网络的学习， 但在预测阶段，P-Net和R-Net仅做人脸检测，不输出关键点位置（因为这时人脸检测都是不准的），关键点位置仅在O-Net中输出。  
5.Bounding box和关键点输出均为归一化后的相对坐标，Bounding Box是相对待检测区域（R-Net和O-Net是相对输入图像），归一化是相对坐标除以检测区域的宽高，关键点坐标是相对Bounding box的坐标，归一化是相对坐标除以Bounding box的宽高，这里先建立起初步的印象，具体可以参看后面准备训练数据部分和预测部分的代码细节。  
