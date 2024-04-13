# 基于街景和路网结构的城市道路出租车碳排放图神经网络估算方法——盲审专用
==**尊敬的盲审老师您好！感谢您的认真负责！这里是论文的实验代码部分，科研不易，望您海涵指正。祝愿您科研工作顺利，身体健康，一帆风顺！**==

> ***简介：**模型使用SAGE或其他图卷积层，并增加图的空间注意力机制，通过结合街景特征和路网结构估算街道机动车碳排放*
>
> **说明：**论文中的数据为了方便本实验的dgl框架使用，直接采用DGLDataset构建数据集，**论文中提到的综合数据集并未放在此处**，因为综合数据集包含了一些本实验未用到的数据，但是本实验所用的三种类型的核心数据都在这里了。

## 1. 实验环境：

（1）PyTorch框架：实验以pytorch框架为基础，安装可参考pytorch官网：[Start Locally | PyTorch](https://pytorch.org/get-started/locally/)

（2）DGL框架：DGL是基于PyTorch的图神经网络框架，安装较为简单，参考官网：[Deep Graph Library (dgl.ai)](https://www.dgl.ai/pages/start.html)

（3）PyTorch的Geometric库：几何图形学深度学习扩展库，这里主要用于方便构建数据集。安装较为复杂，可以参考：[torch_geometric详细安装教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/659091190)

在本实验代码部分，提供了cuda版本和cpu版本两种方式，默认用的cpu版本，因为cpu版本对各电脑兼容性好一些。在SAGE-GSAN.py的代码部分`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`将自动判断是否启用显卡，如果用的cuda，后面代码在部分Tensor与numpy的计算转换时需要更改代码，否则会报错（根据报错提示更改即可）。


## 2. 文件介绍说明：

> 在运行以下文件时，均默认环境已装好。
>
> 请注意并未固定torch.manual_seed()，所以每次训练得到的结果不一定完全相同，如果想要完全相同，给seed固定值即可，比如torch.manual_seed(666666)

1. **SAGE-GSAN.py**文件：此文件是论文相关核心代码文件，实现了论文中的SAGE-GSAN模型，包含了图的空间注意力层，clone下来后可直接运行。

   另外，可以替换模型中的不同图卷积层，来实现文章中的对比实验，如把SAGE 图卷积层替换为GCN即把`self.conv1 = SAGEConv(in_feats, h_feats, 'lstm')`替换为`self.conv1 = GraphConv(in_feats,h_feats)`即可，同时conv2也要替换。

2. **GSAN.py**文件：此文件是单独实现的图空间注意力机制，方便在**SAGE-GSAN.py**文件中直接调用这个文件作为图的空间注意力层使用。

3. **Adjacency relation.csv**文件：路网结构相关信息文件，具体使用方法可参考SAGE-GSAN.py文件中的代码，其中注释写得较为详细

4. **The semantic segmentation features of each road after processing.csv**文件：进行街景的语义分割，然后对每条街道的所有街道特征做平均，然后寻求归一化的结果，该文件的具体用法参考SAGE-GSAN.py文件中的注释，较为清晰。

5. **CGrade.csv**文件：为道路碳排放相关信息文件，具体使用方法可参考SAGE-GSAN.py文件中的注释，写得较为详细。

6. **FeatureToPoint.xlsx**文件：此文件为空间注意力所需的道路坐标信息。

7. **MLP_Predict.py文件**：这是全连接神经网络估算街道碳排放的补充实验模型文件，可以直接运行，代码已经做很好的注释。

8. **dgl_GNNS.py**文件（可忽略这个）：此文件是用SAGE等图卷积网络模型除GAT的代码实现（**注意这个文件不包含图的空间注意力！**）。clone下来项目后可直接运行，通过注释开关可以选择使用哪种图卷积网络层。

9. （可忽略这个）由于GAT的特殊性，GAT和GAT2分别由**dgl_GAT.py和dgl_GAT2_official.py**两个文件分别实现，也可以直接运行。

10. **GNNs_selfAttention.py**文件（可忽略这个）：此文件为普通自注意机制的实验，不是图的空间注意力，因为这种模型基本没有自己的创新，仅仅是组合了图卷积层与普通注意力机制，所以论文并未提到这个，但是由于做了相关实验，所以在这里一并放了进来。