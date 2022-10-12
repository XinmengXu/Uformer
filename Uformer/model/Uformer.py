import torch.nn as nn
import torch 
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(__file__))
from DCCRN.ConvSTFT import ConvSTFT, ConviSTFT

class GLayerNorm2d(nn.Module):
    
    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps 
        self.beta = nn.Parameter(torch.ones([1, in_channel,1,1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel,1,1]))
    
    def forward(self,inputs):
        mean = torch.mean(inputs,[1,2,3], keepdim=True)
        var = torch.var(inputs,[1,2,3], keepdim=True)
        outputs = (inputs - mean)/ torch.sqrt(var+self.eps)*self.beta+self.gamma
        return outputs

class Axial_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=7, stride=1, height_dim=True, inference=False):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads
        
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2)
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size*kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width
            
        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        kqv = self.kqv_conv(x)

        kqv = self.kqv_bn(kqv) # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height), [self.dh // 2, self.dh // 2, self.dh], dim=2)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2, self.kernel_size, self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits) # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        
        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn = torch.matmul(weights, v.transpose(2,3)).transpose(2,3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)
        
        return output

class Axial_Layer_cross(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=7, stride=1, height_dim=True, inference=False):
        super(Axial_Layer_cross, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads
        
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.v_conv = nn.Conv1d(in_channels, self.depth, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm1d(self.depth)

        self.q_conv = nn.Conv1d(in_channels, self.depth // 2, kernel_size=1, bias=False)
        self.q_bn = nn.BatchNorm1d(self.depth // 2)
		
        self.k_conv = nn.Conv1d(in_channels, self.depth // 2, kernel_size=1, bias=False)
        self.k_bn = nn.BatchNorm1d(self.depth // 2)


        self.kq_conv = nn.Conv1d(in_channels, self.depth, kernel_size=1, bias=False)
        self.kq_bn = nn.BatchNorm1d(self.depth)
		
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size*kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x, y):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
            y = y.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width
            y = y.permute(0, 2, 1, 3)  # batch_size, height, depth, width  
			
        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)
        y = y.reshape(batch_size * width, depth, height)
        # Compute q, k, v
        k = self.k_conv(x)
        k = self.k_bn(k) # apply batch normalization on k, q, v
		
        v = self.v_conv(x)
        v = self.kq_bn(v) # apply batch normalization on k, q, v
		
        q = self.q_conv(y)
        q = self.q_bn(q) # apply batch normalization on k, q, v

        kqv = torch.cat([k, q, v], dim = 1)
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height), [self.dh // 2, self.dh // 2, self.dh], dim=2)
        #q = q.reshape(batch_size * width, self.num_heads, self.dh, height)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2, self.kernel_size, self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits) # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        
        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn = torch.matmul(weights, v.transpose(2,3)).transpose(2,3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)
        
        return output

class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x
		
class Model(nn.Module):

    def __init__(self, channel_amp = 1, channel_phase=2):
        super(Model, self).__init__()
        self.stft = ConvSTFT(512, 256, 512, 'hanning', 'complex', True)
        self.istft = ConviSTFT(512, 256, 512, 'hanning', 'complex', True)
		
        self.amp_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_amp, 
                                        kernel_size=[7,1],
                                        padding=(3,0)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                                nn.Conv2d(channel_amp, channel_amp, 
                                        kernel_size=[1,7],
                                        padding=(0,3)
                                    ),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                        )
        self.phase_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_phase, 
                                        kernel_size=[3,5],
                                        padding=(1,2)
                                    ),
                                nn.Conv2d(channel_phase, channel_phase, 
                                        kernel_size=[3,25],
                                        padding=(1, 12)
                                    ),
                        )
        self.amp_conv2 = nn.Sequential(
                        nn.Conv2d(channel_amp, 1, kernel_size=[1, 1]),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                    )
        self.phase_conv2 = nn.Sequential(
                        nn.Conv1d(3,8,kernel_size=[1,1])
                    )

        self.phase_conv5 = nn.Sequential(
                        nn.Conv1d(8, 2, kernel_size=(1,1))
                    )
        self.phase_conv3 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(5,5), padding=(2,2)),
                        GLayerNorm2d(8),
                    )
        self.phase_conv4 = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=(1,25), padding=(0,12)),
                        GLayerNorm2d(8),
                    )

        self.rnn = nn.GRU(
                        257,
                        300,
                        bidirectional=True
                    )
        self.fcs = nn.Sequential(
                    nn.Linear(300*2,600),
                    nn.ReLU(),
                    nn.Linear(600,600),
                    nn.ReLU(),
                    nn.Linear(600,514//2),
                    nn.Sigmoid()
                )					
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        self.SA_time = Axial_Layer(256, height_dim=True)
        self.SA_frequency = Axial_Layer(256, kernel_size=251, height_dim = False)

        self.CA_time_1 = Axial_Layer_cross(256, height_dim=True)
        self.CA_time_2 = Axial_Layer_cross(128, kernel_size=15, height_dim=True)
        self.CA_time_3 = Axial_Layer_cross(64, kernel_size=31, height_dim=True)		
        self.CA_time_4 = Axial_Layer_cross(32, kernel_size=63, height_dim=True)
        self.CA_time_5 = Axial_Layer_cross(16, kernel_size=128, height_dim=True)		
		
		
        self.CA_frequency_1 = Axial_Layer_cross(256, kernel_size=251, height_dim = False)
        self.CA_frequency_2 = Axial_Layer_cross(128, kernel_size=251, height_dim = False)
        self.CA_frequency_3 = Axial_Layer_cross(64, kernel_size=251, height_dim = False)
        self.CA_frequency_4 = Axial_Layer_cross(32, kernel_size=251, height_dim = False)
        self.CA_frequency_5 = Axial_Layer_cross(16, kernel_size=251, height_dim = False)

        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

    def forward(self, x):
        cmp_spec = self.stft(x)
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T]
        cmp_spec = torch.cat([
                                cmp_spec[:,:,:257,:],
                                cmp_spec[:,:,257:,:],
                                ],
                                1)
        # mean = torch.mean(cmp_spec, [1, 2, 3], keepdim = True)
        # std = torch.std(cmp_spec, [1, 2, 3], keepdim = True)
        # cmp_spec = (cmp_spec - mean) / (std + 1e-8)    
        amp_spec = torch.sqrt(
                            torch.abs(cmp_spec[:,0])**2+
                            torch.abs(cmp_spec[:,1])**2,
                        )
        amp_spec = torch.unsqueeze(amp_spec, 1)
        spec = self.amp_conv1(cmp_spec)

        e1 = self.conv_block_1(spec)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)		
 
        b_time = self.SA_time(e5)
        b_frequency = self.SA_frequency(e5)
        b = b_time + b_frequency

        e5_1 = self.CA_time_1(e5, b)
        e5_2 = self.CA_frequency_1(e5, b)
        d = self.tran_conv_block_1(torch.cat([b, e5_1], 1))
        e4_1 = self.CA_time_2(e4, d)
        e4_2 = self.CA_frequency_2(e4, d)
        d = self.tran_conv_block_2(torch.cat([d, e4_1], 1))
        e3_1 = self.CA_time_3(e3, d)
        e3_2 = self.CA_frequency_3(e3, d)		
        d = self.tran_conv_block_3(torch.cat([d, e3_1], 1))
        e2_1 = self.CA_time_4(e2, d)
        e2_2 = self.CA_frequency_4(e2, d)
        d = self.tran_conv_block_4(torch.cat([d, e2_1], 1))
        e1_1 = self.CA_time_5(e1, d)
        e1_2 = self.CA_frequency_5(e1, d)
        d = self.tran_conv_block_5(torch.cat([d, e1_1], 1))

        spec = torch.transpose(d, 1,3)
        #print(spec.size())
        B, T, D, C = spec.size()
        spec = torch.reshape(spec, [B, T, D*C])
        spec = self.rnn(spec)[0]
        spec = self.fcs(spec)
        
        spec = torch.reshape(spec, [B,T,D,1]) 
        spec = torch.transpose(spec, 1,3)

	
        phase_pro = self.phase_conv1(cmp_spec)			
        phase_input = torch.cat([phase_pro, self.amp_conv2(d)], dim = 1)
      
        phase_input = self.phase_conv2(phase_input)	
        p1 = self.phase_conv3(phase_input)
        p1 = self.phase_conv4(p1)
		
        p2 = self.phase_conv3(p1 + phase_input)
        p2 = self.phase_conv4(p2)
		
        p3 = self.phase_conv3(p2 + p1)
        p3 = self.phase_conv4(p3)

        p5 = self.phase_conv5(p3)
        p5 = phase_pro + p5
        p5 = p5/(torch.sqrt(
                            torch.abs(p5[:,0])**2+
                            torch.abs(p5[:,1])**2)
                        +1e-8).unsqueeze(1)
        est_spec = amp_spec * d * p5
        est_spec = torch.cat([est_spec[:,0], est_spec[:,1]], 1)
        est_wav = self.istft(est_spec, None)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav

    def loss(self, est, labels, mode='Mix'):
        '''
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        '''
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels,1)
            if est.dim() == 3:
                est = torch.squeeze(est,1)
            return -si_snr(est, labels)         
        elif mode == 'Mix':
            b, d, t = est.size()
            gth_cspec = self.stft(labels)
            est_cspec = est  
            gth_mag_spec = torch.sqrt(
                                    gth_cspec[:, :self.feat_dim, :]**2
                                    +gth_cspec[:, self.feat_dim:, :]**2 + 1e-8
                               )
            est_mag_spec = torch.sqrt(
                                    est_cspec[:, :self.feat_dim, :]**2
                                    +est_cspec[:, self.feat_dim:, :]**2 + 1e-8
                                )
            
            # power compress 
            gth_cprs_mag_spec = gth_mag_spec**0.3
            est_cprs_mag_spec = est_mag_spec**0.3
            amp_loss = F.mse_loss(
                                gth_cprs_mag_spec, est_cprs_mag_spec
                            )*d
            compress_coff = (gth_cprs_mag_spec/(1e-8+gth_mag_spec)).repeat(1,2,1)
            phase_loss = F.mse_loss(
                                gth_cspec*compress_coff,
                                est_cspec*compress_coff
                            )*d
            
            all_loss = amp_loss*0.5 + phase_loss*0.5
            return all_loss, amp_loss, phase_loss

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data
def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def test_selfattention1():
    torch.manual_seed(20)

    inputs = torch.randn([1, 64000])
    net = Model()
    
    out1, out2 = net(inputs)
    print(out2.shape)
	
if __name__ == '__main__':
    test_selfattention1()