import math
import torch


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x


class MemoryBank:
    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k

    """
        全局匹配
        输入：memory key 和 query key 
        输出：affinity
    """
    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2) # B, NE, 1 
        ab = mk.transpose(1, 2) @ qk # B, NE, HW
 
        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW  (top_k)

        """
        不用 top_k 在 davis2017-val 上: 83.3
        import torch.nn.functional as F
        affinity = F.softmax(affinity, dim=1) 
        """
        import torch.nn.functional as F
        # affinity = F.softmax(affinity, dim=1) 

        return affinity

    """
        read
        输入：affinity 和 memory value 
        输出：readout
    """
    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    """
        整个 memory read 过程
        输入：query key 
        操作：1.计算 affinity 2.计算 readout 
        输出：readout (k, CV, h, w) 
    """
    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        mk_test = mk.view(1, self.CK, -1, h, w)
        mv_test = mv.view(k, self.CV, -1, h, w)
        _, _, T, _, _ = mk_test.shape
        
        # first + last_4
        if T <= 5:
            mk_range = range(T)
        else:
            mk_range = [0] + list(range(-4, 0))
        
        # first + ave_4
        # if T <= 5:
        #     mk_range = range(T)
        # else:
        #     num_remaining_t = T - 1
        #     mk_range = [0] + torch.linspace(1, num_remaining_t, steps=4, dtype=torch.long).tolist()
        
        mk_select = mk_test[:, :, mk_range, :, :]
        mv_select = mv_test[:, :, mk_range, :, :]

        mk_select = mk_select.flatten(start_dim=2)
        mv_select = mv_select.flatten(start_dim=2)

        affinity = self._global_matching(mk, qk)
        # affinity = self._global_matching(mk_select, qk)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)
        # readout_mem = self._readout(affinity.expand(k,-1,-1), mv_select)

        return readout_mem.view(k, self.CV, h, w)
    
    # 记忆存储，将记忆帧的 key 和 value 加入记忆池
    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)