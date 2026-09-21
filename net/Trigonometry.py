# Trigonometry.py 提供 GerNA-Bind pairwise 模块里的几何更新层。
# 核心输入通常是 z [B, I, J, C]：I 是 RNA base/token 数，J 是分子 atom 数，C 是 pair embedding 通道。
# protein_pair [B, I, I, C] 表示 RNA 内部几何关系；compound_pair [B, J, J, C] 表示分子内部几何关系。
import torch
import torch.nn as nn
import torch.nn.functional as F

# 将坐标两两距离离散化成 16-bin one-hot。
# 输入 d: [B, N, 3]，输出: [B, N, N, 16]，用于告诉三角模块 RNA/RNA 或 atom/atom 的内部几何。
def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    # pair_dis[b, i, k] 是同一样本里第 i 个点和第 k 个点的欧氏距离。
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    # 连续距离按 bin_size 分桶；bin_min=-1 让 0 距离落到第一个有效桶附近。
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    return pair_dis_one_hot

# 旧版三角更新实现；当前 net/model.py 实际使用的是 TriangleProteinToCompound_v3。
# 保留它主要是历史代码，阅读主流程可以先跳过 v2。
class TriangleProteinToCompound_v2(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = nn.Linear(embedding_channels, c)
        self.gate_linear2 = nn.Linear(embedding_channels, c)

        self.linear1 = nn.Linear(embedding_channels, c)
        self.linear2 = nn.Linear(embedding_channels, c)

        self.ending_gate_linear = nn.Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = nn.Linear(c, embedding_channels)
    # z: [B, I, J, C]，当前 RNA i - atom j 的 pair embedding。
    # protein_pair: [B, I, I, C]，RNA i - RNA k 的内部距离/几何 embedding。
    # compound_pair: [B, J, J, C]，atom j - atom k 的内部距离/几何 embedding。
    # z_mask: [B, I, J, 1]，padding 位置为 0。
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)

        #print(z.shape,z_mask.shape)
        # ab1/ab2 是两条三角路径上的 RNA-molecule pair 信息；gate 决定每个通道通过多少。
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        # RNA 内部 pair 和分子内部 pair 也先做 gated projection，统一到中间维度 c。
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()
        #print(protein_pair.shape, ab1.shape)
        #print(compound_pair.shape, ab2.shape)
        block1 = torch.einsum("bic,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjc->bijc", ab2, compound_pair)
        # print(g.shape, block1.shape, block2.shape)
        # 合并两条三角路径，投回原通道 C；这里返回的是更新量，model.py 里会用残差 z = z + update。
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

# 三角更新层：用 RNA-RNA 内部关系和 molecule-molecule 内部关系更新 RNA-molecule pair 表征 z。
# 直观上，预测 pair (i,j) 时会参考所有 (i,k)/(k,j) 或 (i,k)/(j,k) 相关路径，减少几何上不一致的 pairwise 预测。
class TriangleProteinToCompound_v3(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = nn.Linear(embedding_channels, c)
        self.gate_linear2 = nn.Linear(embedding_channels, c)

        self.linear1 = nn.Linear(embedding_channels, c)
        self.linear2 = nn.Linear(embedding_channels, c)

        self.ending_gate_linear = nn.Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = nn.Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
 
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()
        #print(protein_pair.shape, ab1.shape)
        #print(compound_pair.shape, ab2.shape)
        #print("protein_pair shape:", protein_pair.shape)
        #print("ab1 shape:", ab1.shape)
        # RNA 侧三角路径：对目标 pair (i,j)，枚举 RNA k，聚合 RNA(i,k) * pair(k,j)。
        # einsum: protein_pair [B,I,I,c] 与 ab1 [B,I,J,c] 在 k 维相乘求和，输出 block1 [B,I,J,c]。
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        # 分子侧三角路径：对目标 pair (i,j)，枚举 atom k，聚合 pair(i,k) * atom_pair(j,k)。
        # 输出 block2 [B,I,J,c]，让同一 RNA base 对不同 atom 的预测互相通信。
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)
        # print(g.shape, block1.shape, block2.shape)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z


# 普通多头 self-attention 实现；当前 GerNA 主路径没有直接使用这个类。
# 实际主路径使用下面的 TriangleSelfAttentionRowWise。
class Self_Attention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads=8,drop_rate=0.5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None):
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dp(attention_probs)
        if attention_weight is not None:
            attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
            # if self.training:
            #     top_mask = (attention_weight_sorted_sorted<np.random.randint(28,45))
            # else:
            top_mask = (attention_weight_sorted_sorted<32)
            attention_probs = attention_probs * top_mask
            # attention_probs = attention_probs * attention_weight
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)
        # print(attention_probs.shape,v.shape)
        # attention_probs = self.dp(attention_probs)
        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.ln(outputs)
        return outputs

# 对 z [B, I, J, C] 做 row-wise attention：固定 RNA i，在 molecule atom 维度 J 上做 self-attention。
# 作用是让同一个 RNA base 对不同 atom 的 pair 表征互相交流。
class TriangleSelfAttentionRowWise(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.dp = nn.Dropout(drop_rate)
        # self.ln = nn.LayerNorm(hidden_size)

        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        # self.layernorm_c = torch.nn.LayerNorm(c)

        self.linear_q = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        # self.b = Linear(embedding_channels, h, bias=False)
        self.g = nn.Linear(embedding_channels, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    # z: [B, I, J, C]；z_mask: [B, I, J]。
    # 对每个样本 b、每个 RNA i，计算 J 个 atom 之间的多头注意力。
    def forward(self, z, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        # new_z = torch.zeros(z.shape, device=z.device)
        z_i = z
        # mask reshape 成 [B, I, 1, 1, J]，用于屏蔽 padding atom。
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        # q, k, v of shape b, j, h, c
        # q/k/v: [B, I, J, heads, head_dim]。
        q = self.reshape_last_dim(self.linear_q(z_i)) #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z_i))
        v = self.reshape_last_dim(self.linear_v(z_i))
        # 固定 b 和 RNA i，对 atom q 与 atom k 做注意力打分。
        # logits: [B, I, heads, J(query atom), J(key atom)]。
        logits = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        weights = nn.Softmax(dim=-1)(logits)
        # weights of shape b, h, j, j
        # attention_probs = self.dp(attention_probs)
        # 用 attention 权重聚合 value，输出 [B, I, J, heads, head_dim]。
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        g = self.reshape_last_dim(self.g(z_i)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        # output of shape b, j, embedding.
        # z[:, i] = output
        z = output
        # print(g.shape, block1.shape, block2.shape)
        # 多头拼回 C 维并乘 mask；返回更新量，model.py 里同样做残差加回。
        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        return z


# Transformer 风格的前馈层 FFN：对每个 pair embedding 独立做 C -> 4C -> C 的非线性变换。
# 输入输出形状都保持 [B, I, J, C]。
class Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = nn.Linear(embedding_channels, n*embedding_channels)
        self.linear2 = nn.Linear(n*embedding_channels, embedding_channels)
    # Transition 不混合不同 pair，只在最后一维通道上做 MLP 更新。
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z