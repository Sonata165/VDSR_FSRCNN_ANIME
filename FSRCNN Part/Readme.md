## FSRCNN超分辨率 + NLM去噪

### 模块

| 模块        | 作用                                 |
| ----------- | ------------------------------------ |
| Constants   | 定义常量                             |
| Criteria    | 计算PSNR和SSIM的实现代码             |
| DataPrepare | 准备训练和测试数据                   |
| Denoise     | 对网络输出进行去噪                   |
| Models      | 网络结构定义                         |
| Test        | 将测试集输入训练好的网络得到网络输出 |
| Train       | 使用训练集训练网络                   |
| Utils       | 工具函数和类                         |

### 目录

| 目录    | 作用                       |
| ------- | -------------------------- |
| data    | 放数据                     |
| figure  | 存放对patch output的可视化 |
| log     | 训练的Tensorboard日志      |
| models  | 存放训练好的模型           |
| results | 报告中的结果               |

