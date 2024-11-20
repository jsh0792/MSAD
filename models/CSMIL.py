import torch
import torch.nn as nn
"""
Model definition of DeepAttnMISL

If this work is useful for your research, please consider to cite our papers:

[1] "Whole Slide Images based Cancer Survival Prediction using Attention Guided Deep Multiple Instance Learning Networks"
Jiawen Yao, XinliangZhu, Jitendra Jonnagaddala, NicholasHawkins, Junzhou Huang,
Medical Image Analysis, Available online 19 July 2020, 101789

[2] "Deep Multi-instance Learning for Survival Prediction from Whole Slide Images", In MICCAI 2019

"""

class CSMIL(nn.Module):
    """
    Deep AttnMISL Model definition
    """

    def __init__(self, cluster_num=6, input_dim=768, n_class=4):
        super(CSMIL, self).__init__()
        self.embedding_net = nn.Sequential(
                                    nn.Conv2d(input_dim, 64, 1),
                                    # nn.Linear(input_dim, 64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1,1))
                                     )

        self.res_attention = nn.Sequential(
            nn.Conv2d(64, 32, 1),  # V
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 32), # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, n_class),
            nn.Sigmoid()
        )
        self.cluster_num = cluster_num


    def masked_softmax(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / (mask+1e-5))
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)


    def forward(self,  **kwargs):
        x_path_5x = kwargs['x_path_5x']
        x_path_10x = kwargs['x_path_10x']
        x_path_20x = kwargs['x_path_20x']
        res = []
        for i in range(self.cluster_num):

            output1 = self.embedding_net(x_path_5x[i].unsqueeze(0).unsqueeze(2).unsqueeze(3))   # torch.size([6, 64, 1, 1])
            output2 =  self.embedding_net(x_path_10x[i].unsqueeze(0).unsqueeze(2).unsqueeze(3))
            output3 =  self.embedding_net(x_path_20x[i].unsqueeze(0).unsqueeze(2).unsqueeze(3))
            
            output = torch.cat([output1, output2, output3],2)
            res_attention = self.res_attention(output).squeeze(-1)

            final_output = torch.matmul(output.squeeze(-1), torch.transpose(res_attention,2,1)).squeeze(-1)
            res.append(final_output)
            
        h = torch.cat(res)

        b = h.size(0)
        c = h.size(1)

        h = h.view(b, c)
        A = self.attention(h)
        A = torch.transpose(A, 1, 0)  # KxN

        A = self.masked_softmax(A, mask=None)

        M = torch.mm(A, h)  # KxL

        logits = self.fc6(M)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return logits, hazards, S, None

if __name__ == '__main__':
    # 假设你的模型已经定义为 DeepAttnMIL_Surv
    cluster_num = 3  # 例如，3个聚类
    batch_size = 5   # 例如，5个样本
    num_items = 10   # 假设每个样本有10个项
    
    device = 'cuda'
    path_5x = torch.rand(6,1024).to(device)
    path_10x = torch.rand(6,1024).to(device)
    path_20x = torch.rand(6,1024).to(device)
    
    # 创建模型实例
    model = CSMIL(cluster_num).to("cuda")

    # # 创建输入张量 x
    # x = torch.randn(6, batch_size, 3, 2048).to("cuda")  # 随机生成输入，形状为 (cluster_num, batch_size, 3, 2048)

    # # 创建掩码
    # mask = torch.randint(0, 2, (batch_size, num_items)).float().to("cuda")  # 随机生成掩码

    # 测试模型
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(x_path_5x=path_5x, x_path_10x=path_10x, x_path_20x=path_20x)  # 前向传播
        print("Model Output:", output)