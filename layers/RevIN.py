import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x



# import torch
# import torch.nn as nn
# import torch.fft as fft

# class RevIN(nn.Module):
#     def __init__(self, num_features: int, eps=1e-5, affine=True, 
#                  subtract_last=False, 
#                  use_periodic_mean=True,
#                  pred_len=96,
#                  period_method="fft"):   # "repeat" | "mirror" | "fft"
#         super(RevIN, self).__init__()
        
#         self.num_features = num_features
#         self.eps = eps
#         self.affine = affine
#         self.subtract_last = subtract_last
        
#         # 新增
#         self.use_periodic_mean = use_periodic_mean
#         self.pred_len = pred_len
#         self.period_method = period_method
        
#         if self.affine:
#             self._init_params()

#     def forward(self, x, mode:str):
#         if mode == 'norm':
#             self._get_statistics(x)
#             x = self._normalize(x)
#         elif mode == 'denorm':
#             x = self._denormalize(x)
#         else: 
#             raise NotImplementedError
#         return x

#     def _init_params(self):
#         self.affine_weight = nn.Parameter(torch.ones(self.num_features))
#         self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

#     # ========================= 核心修改 =========================
#     def _get_statistics(self, x):
#         B, T, C = x.shape

#         if self.subtract_last:
#             self.last = x[:, -1:, :]
        
#         # 👉 用周期均值替代普通均值
#         if self.use_periodic_mean:
#             full_seq = self._reconstruct_period(x)   # (B, T+pred_len, C)
#             dim2reduce = tuple(range(1, full_seq.ndim-1))
#             self.mean = torch.mean(full_seq, dim=dim2reduce, keepdim=True).detach()
#         else:
#             dim2reduce = tuple(range(1, x.ndim-1))
#             self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

#         # std 还是用原始数据（更稳定）
#         dim2reduce = tuple(range(1, x.ndim-1))
#         self.stdev = torch.sqrt(
#             torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
#         ).detach()

#     # ========================= 周期补全 =========================
#     def _reconstruct_period(self, x):
#         if self.period_method == "repeat":
#             return self._repeat_extend(x)
#         elif self.period_method == "mirror":
#             return self._mirror_extend(x)
#         elif self.period_method == "fft":
#             return self._fft_extend(x)
#         else:
#             raise NotImplementedError

#     # 方法1：重复
#     def _repeat_extend(self, x):
#         return torch.cat([x, x[:, :self.pred_len, :]], dim=1)

#     # 方法2：镜像
#     def _mirror_extend(self, x):
#         return torch.cat([x, torch.flip(x, dims=[1])[:, :self.pred_len, :]], dim=1)

#     # ⭐ 方法3：FFT补全（推荐）
#     def _fft_extend(self, x):
#         # 1. 去均值
#         mean = x.mean(dim=1, keepdim=True)
#         x = x - mean

#         # 2. FFT
#         Xf = torch.fft.rfft(x, dim=1)

#         # 3. 只保留低频（去噪）
#         k = 5
#         Xf[:, k:, :] = 0

#         # 4. IFFT（但不改长度！）
#         x_smooth = torch.fft.irfft(Xf, n=x.shape[1], dim=1)

#         # 5. 周期复制
#         x_new = torch.cat([x_smooth, x_smooth[:, :self.pred_len, :]], dim=1)

#         return x_new + mean

#     # ========================= 原逻辑 =========================
#     def _normalize(self, x):
#         if self.subtract_last:
#             x = x - self.last
#         else:
#             x = x - self.mean

#         x = x / self.stdev

#         if self.affine:
#             x = x * self.affine_weight
#             x = x + self.affine_bias

#         return x

#     def _denormalize(self, x):
#         if self.affine:
#             x = x - self.affine_bias
#             x = x / (self.affine_weight + self.eps*self.eps)

#         x = x * self.stdev

#         if self.subtract_last:
#             x = x + self.last
#         else:
#             x = x + self.mean

#         return x




# import torch
# import torch.nn as nn
# import torch.fft as fft


# class RevIN(nn.Module):
#     def __init__(
#         self,
#         num_features: int,
#         eps=1e-5,
#         affine=True,
#         subtract_last=False,
#         use_periodic_mean=True,
#         pred_len=96,
#         period_method="ode",  # "repeat" | "mirror" | "fft" | "ode" | "neural_ode"
#         ode_hidden=32
#     ):
#         super().__init__()

#         self.num_features = num_features
#         self.eps = eps
#         self.affine = affine
#         self.subtract_last = subtract_last

#         self.use_periodic_mean = use_periodic_mean
#         self.pred_len = pred_len
#         self.period_method = period_method

#         # ODE parameters
#         self.ode_hidden = ode_hidden

#         if affine:
#             self._init_params()

#         # lazy init for ODE model
#         self.ode_func = None

#     # ===================== init affine =====================
#     def _init_params(self):
#         self.affine_weight = nn.Parameter(torch.ones(self.num_features))
#         self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

#     # ===================== forward =====================
#     def forward(self, x, mode: str):
#         if mode == "norm":
#             self._get_statistics(x)
#             return self._normalize(x)
#         elif mode == "denorm":
#             return self._denormalize(x)
#         else:
#             raise NotImplementedError

#     # ===================== statistics =====================
#     def _get_statistics(self, x):
#         B, T, C = x.shape

#         if self.subtract_last:
#             self.last = x[:, -1:, :]

#         # mean
#         if self.use_periodic_mean:
#             full_seq = self._reconstruct_period(x)
#             self.mean = full_seq.mean(dim=(1, 2), keepdim=True).detach()
#         else:
#             self.mean = x.mean(dim=(1, 2), keepdim=True).detach()

#         # std (always original)
#         self.stdev = torch.sqrt(
#             torch.var(x, dim=(1, 2), keepdim=True, unbiased=False) + self.eps
#         ).detach()

#     # ===================== reconstruction router =====================
#     def _reconstruct_period(self, x):
#         if self.period_method == "repeat":
#             return self._repeat_extend(x)
#         elif self.period_method == "mirror":
#             return self._mirror_extend(x)
#         elif self.period_method == "fft":
#             return self._fft_extend(x)
#         elif self.period_method == "ode":
#             return self._ode_extend(x)
#         else:
#             raise NotImplementedError

#     # ===================== repeat =====================
#     def _repeat_extend(self, x):
#         return torch.cat([x, x[:, :self.pred_len, :]], dim=1)

#     # ===================== mirror =====================
#     def _mirror_extend(self, x):
#         return torch.cat([x, torch.flip(x, dims=[1])[:, :self.pred_len, :]], dim=1)

#     # ===================== FFT extend (denoise + repeat) =====================
#     def _fft_extend(self, x):
#         mean = x.mean(dim=1, keepdim=True)
#         x0 = x - mean

#         Xf = torch.fft.rfft(x0, dim=1)

#         # low-frequency filter
#         k = 5
#         Xf[:, k:, :] = 0

#         x_smooth = torch.fft.irfft(Xf, n=x.shape[1], dim=1)

#         x_ext = torch.cat(
#             [x_smooth, x_smooth[:, :self.pred_len, :]],
#             dim=1
#         )

#         return x_ext + mean

#     # ===================== ODE extend (Euler dynamics) =====================
#     def _ode_extend(self, x):
#         B, T, C = x.shape
#         device = x.device

#         mean = x.mean(dim=1, keepdim=True)
#         x0 = x - mean

#         # init ODE function (learnable)
#         if self.ode_func is None:
#             self.ode_func = nn.Sequential(
#                 nn.Linear(C, self.ode_hidden),
#                 nn.Tanh(),
#                 nn.Linear(self.ode_hidden, C)
#             ).to(device)

#         # initial state
#         x_t = x0[:, -1:, :]
#         outputs = []

#         for _ in range(self.pred_len):
#             dx = self.ode_func(x_t)
#             x_t = x_t + dx
#             outputs.append(x_t)

#         x_future = torch.cat(outputs, dim=1)

#         return torch.cat([x0, x_future], dim=1) + mean

#     # ===================== normalize =====================
#     def _normalize(self, x):
#         if self.subtract_last:
#             x = x - self.last
#         else:
#             x = x - self.mean

#         x = x / (self.stdev + self.eps)

#         if self.affine:
#             x = x * self.affine_weight + self.affine_bias

#         return x

#     # ===================== denormalize =====================
#     def _denormalize(self, x):
#         if self.affine:
#             x = (x - self.affine_bias) / (self.affine_weight + self.eps)

#         x = x * self.stdev

#         if self.subtract_last:
#             x = x + self.last
#         else:
#             x = x + self.mean

#         return x




# import torch
# import torch.nn as nn


# class RevIN(nn.Module):
#     def __init__(self, num_features: int, eps=1e-5, affine=True, pred_len=96,subtract_last=False):
#         super(RevIN, self).__init__()

#         self.num_features = num_features
#         self.eps = eps
#         self.affine = affine
#         self.subtract_last = subtract_last

#         if self.affine:
#             self._init_params()

#         # =========================
#         # Channel dependency module
#         # =========================
#         self.mean_mixer = nn.Sequential(
#             nn.Linear(num_features, num_features),
#             nn.ReLU(),
#             nn.Linear(num_features, num_features)
#         )

#     def forward(self, x, mode: str):
#         if mode == 'norm':
#             self._get_statistics(x)
#             x = self._normalize(x)
#         elif mode == 'denorm':
#             x = self._denormalize(x)
#         else:
#             raise NotImplementedError
#         return x

#     def _init_params(self):
#         # affine parameters per channel
#         self.affine_weight = nn.Parameter(torch.ones(self.num_features))
#         self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

#     # =========================================================
#     # 🔥 Key improvement: channel-dependent statistics modeling
#     # =========================================================
#     def _get_statistics(self, x):
#         """
#         x: [B, L, C]
#         """

#         # ---- basic per-channel stats ----
#         mean = x.mean(dim=1, keepdim=False)  # [B, C]
#         var = x.var(dim=1, keepdim=False, unbiased=False)

#         # ---- channel dependency modeling ----
#         # mix information across channels
#         mixed_mean = self.mean_mixer(mean) + mean  # [B, C]

#         # reshape for broadcasting
#         self.mean = mixed_mean.unsqueeze(1).detach()  # [B,1,C]
#         self.stdev = torch.sqrt(var + self.eps).unsqueeze(1).detach()

#         if self.subtract_last:
#             self.last = x[:, -1, :].unsqueeze(1).detach()

#     # =========================================================
#     def _normalize(self, x):
#         if self.subtract_last:
#             x = x - self.last
#         else:
#             x = x - self.mean

#         x = x / self.stdev

#         if self.affine:
#             x = x * self.affine_weight
#             x = x + self.affine_bias

#         return x

#     # =========================================================
#     def _denormalize(self, x):
#         if self.affine:
#             x = x - self.affine_bias
#             x = x / (self.affine_weight + self.eps * self.eps)

#         x = x * self.stdev

#         if self.subtract_last:
#             x = x + self.last
#         else:
#             x = x + self.mean

#         return x









import torch
import torch.nn as nn

class FlowAffine(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, inverse=False):
        B, L, C = x.shape
        shape = [1, 1, C]  # 正确维度，适配 [B, L, C]
        
        g = torch.clamp(self.log_gamma, -4, 4)
        g = g.view(shape).expand_as(x)
        beta = self.beta.view(shape).expand_as(x)
        
        if not inverse:
            return x * torch.exp(g) + beta
        else:
            return (x - beta) * torch.exp(-g)

class RevIN_FlowBlock(nn.Module):
    def __init__(self, num_features: int, block_len=2, eps=1e-5, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.block_len = block_len
        self.eps = eps
        self.subtract_last = subtract_last

        # ===================== 核心改动 =====================
        # 这里先不初始化，forward 里根据序列长度动态创建每块独立 affine
        # ====================================================
        self.flow_affines = None  # 每块独立的 FlowAffine 列表

    def split_into_blocks(self, x):
        B, L, C = x.shape
        blocks = []
        start = 0
        while start < L:
            end = min(start + self.block_len, L)
            block = x[:, start:end, :]
            blocks.append(block)
            start = end
        return blocks, L

    def merge_blocks(self, blocks, L):
        return torch.cat(blocks, dim=1)

    def _get_block_stats(self, block):
        if self.subtract_last:
            last = block[:, -1:, :].detach()
            return last, None
        else:
            mean = torch.mean(block, dim=1, keepdim=True).detach()
            var = torch.var(block, dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.sqrt(var + self.eps).detach()
            return mean, stdev

    def _normalize_block(self, block, affine_index):
        mean, stdev = self._get_block_stats(block)
        if self.subtract_last:
            x = block - mean
        else:
            x = block - mean
            x = x / stdev
        
        # 每块使用独立的 FlowAffine
        x = self.flow_affines[affine_index](x, inverse=False)
        return x, mean, stdev

    def _denormalize_block(self, block, mean, stdev, affine_index):
        # 每块使用独立的 FlowAffine
        x = self.flow_affines[affine_index](block, inverse=True)
        
        if self.subtract_last:
            x = x + mean
        else:
            x = x * stdev
            x = x + mean
        return x

    def forward(self, x, mode: str):
        B, L, C = x.shape
        blocks, total_len = self.split_into_blocks(x)
        num_blocks = len(blocks)

        # ===================== 核心：动态创建每块独立 FlowAffine =====================
        if self.flow_affines is None or len(self.flow_affines) != num_blocks:
            self.flow_affines = nn.ModuleList([
                FlowAffine(self.num_features).to(x.device) 
                for _ in range(num_blocks)
            ])
        # ==========================================================================

        if mode == 'norm':
            norm_blocks = []
            self.stats = []
            for i, blk in enumerate(blocks):
                norm_blk, mean, stdev = self._normalize_block(blk, affine_index=i)
                norm_blocks.append(norm_blk)
                self.stats.append((mean, stdev))
            return self.merge_blocks(norm_blocks, total_len)

        elif mode == 'denorm':
            denorm_blocks = []
            for i, blk in enumerate(blocks):
                mean, stdev = self.stats[i]
                denorm_blk = self._denormalize_block(blk, mean, stdev, affine_index=i)
                denorm_blocks.append(denorm_blk)
            return self.merge_blocks(denorm_blocks, total_len)

        else:
            raise NotImplementedError("mode only support 'norm' or 'denorm'")