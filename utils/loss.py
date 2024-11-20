import torch

class NLLSurvLoss(object):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))         
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def rankloss(S, Y, c, event_time):
    # S (n, 4)      Y (n, 1)
    risk = torch.gather(S, 1, Y)
    # for idx, i in enumerate(risk):
    #     if c[idx] == 1:
    #         risk[idx] = S[idx][-1]
    risk = -risk
    risk_matrix = risk - risk.T
    Comp = torch.zeros_like(risk_matrix)
    # 计算comparable pairs矩阵 C
    for i, row in enumerate(risk_matrix):
        for j in range(len(row)):
            if c[i] == 0 and event_time[i] < event_time[j]:
                Comp[i][j] = 1

    risk_matrix = torch.clamp(1-risk_matrix,min=0)
    result = (Comp * risk_matrix).sum()
    return result


import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable


# c refers to the number of classes
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha)
    S_beta = torch.sum(beta)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha))
    lnB_uni = torch.sum(torch.lgamma(beta)) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0)) + lnB + lnB_uni
    return kl


# Yj refers to time of death
# dim refers to dimensions of alpha
def hazard_loss_(alpha, label, Yj, device):
    S = torch.sum(alpha)
    # label = torch.zeros(dim).to(device)
    # label[Yj] = 1
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)))
    return A


def surv_loss_(alpha, L, dim, Yj, device):
    mu = (torch.log(alpha).T - torch.mean(torch.log(alpha))).T
    var = (((1.0 / alpha) * (1 - (2.0 / dim))).T +
                             (1.0 / (dim * dim)) * torch.sum(1.0 / alpha)).T
    var = torch.diag(var)
    # sample
    m = MultivariateNormal(mu, var)
    eps = Variable(m.sample((L, )), requires_grad=True)
    # reparameterization
    m_sample = F.softmax(torch.sqrt(torch.diagonal(var))*eps + mu)
    loss_all = 0
    for u in range(0, Yj):
        label = torch.zeros(dim).to(device)
        label[u] = 1
        # Multinomial
        loss_all += torch.log(1-label*m_sample).sum()/L
    return -loss_all


# Total loss function
# global_step and annealing_step are used to control the increase of λ as the number of training rounds increases
# The definitions of alpha and p are the same as in the paper
# L refers to the number of samples
def ce_loss(evidence, c, global_step, annealing_step, L, censor, Yj, device):

    alpha = evidence + 1
    S = torch.sum(alpha)
    p = alpha / S

    #Yj = torch.argmax(p)  # Time of death
    E = alpha - 1
    label = torch.zeros(c).to(device)
    label[Yj] = 1
    hazard = hazard_loss_(alpha, label, Yj, device)
    annealing_coef = min(1, global_step / annealing_step)  # λt
    alp = E * (1 - label) + 1
    KL_ = annealing_coef * KL(alp, c)  # KL divergence
    if censor == 1:
        survive = surv_loss_(alpha, L, c, Yj+1, device)
        loss = censor*survive
    else:
        survive = surv_loss_(alpha, L, c, Yj, device)
        loss = (1-censor)*(survive + hazard)
    return loss + KL_


def DS_Combin(alpha, class_num):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """
    def DS_Combin_two(alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = class_num/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, class_num, 1), b[1].view(-1, 1, class_num))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = class_num / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    for v in range(len(alpha)-1):
        if v==0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
    return alpha_a

if __name__ == '__main__':
    evidence_1 = torch.tensor([0.55, 0.13, 0.47, 0.63]).cuda()
    evidence_2 = torch.tensor([0.12, 0.13, 0.65, 0.23]).cuda()
    censor = 1
    c = 4
    ce_mm = ce_loss(evidence_1, c, 5, 10, 100, censor)
    ce_gene = ce_loss(evidence_2, c, 5, 10, 100, censor)

    alpha_1 = evidence_1 + 1
    alpha_2 = evidence_2 + 1
    alpha_list = [alpha_1, alpha_2]
    alpha_aggr = DS_Combin(alpha_list)

    ce_aggr = ce_loss(alpha_aggr-1, c, 5, 10, 100, censor)