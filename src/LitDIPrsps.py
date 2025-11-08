"""
This script was modified from a script provided by Czerkawski et al., 
wich you can get from https://github.com/strath-ai/satellite-cloud-removal-dip.
"""

import pytorch_lightning as pl
from .DeepImagePrior.SkipNetwork import *
from .Dataset import *

import torch
import torch.nn.functional as F

class LitDIPrsps(pl.LightningModule):
    def __init__(self,
                 target_channels = 3,
                 latent_seed = 'meshgrid',
                 sigmoid_output = True,
                 lr = 2e-2,
                 regularization_noise_std = 1e-1,
                 epoch_steps = 1000,
                 logging = False,                
                 # ---- range prior ----
                 cluster_loss_weight: float = 0.0,  
                 cluster_min_count: int = 12,       
                 cluster_q_low: float = 0.05,       
                 cluster_q_high: float = 0.95,      
                 # ---- shape prior ----
                 shape_loss_weight: float = 0.0,     
                 tv_in_weight: float = 1.0,          
                 edge_align_weight: float = 1.0,     
                 edge_tau: float = 0.03,             
                 edge_dilate: int = 1                                    
                ):
        super().__init__()
        
        self.input = None        
        self.target = None
        self.mask = None
        self.latent_seed = latent_seed
        self.sigmoid_output = sigmoid_output
        self.logging = logging
        self.epoch_steps = epoch_steps
        self.target_channels = target_channels
        self.regularization_noise_std = regularization_noise_std
        self.lr = lr

        self.cluster_loss_weight = float(cluster_loss_weight)
        self.cluster_min_count   = int(cluster_min_count)
        self.cluster_q_low       = float(cluster_q_low)
        self.cluster_q_high      = float(cluster_q_high)
        self.cluster_map         = None  
        self._cluster_stats      = None 

        self.shape_loss_weight = float(shape_loss_weight)
        self.tv_in_weight      = float(tv_in_weight)
        self.edge_align_weight = float(edge_align_weight)
        self.edge_tau          = float(edge_tau)
        self.edge_dilate       = int(edge_dilate)
        self._Bmask            = None    
        self._Win              = None  

        if isinstance(self.latent_seed, list) or isinstance(self.latent_seed, tuple):
            self.input_depth = self.latent_seed[1]
        elif self.latent_seed == 'meshgrid':
            self.input_depth = 2
        else:
            self.input_depth = np.sum(self.target_channels)

        self.init_model()           
       
    def image_check(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 0)  
        if np.argmin(x.shape[1:]) == 2:
            x = x.permute(0, 3, 1, 2)
        return x.to(self.device)
    
    def init_model(self):
        self.model = SkipNetwork(in_channels = self.input_depth,
                                 out_channels = int(np.sum(self.target_channels)),
                                 hidden_dims_down = [16, 32, 64, 128, 128, 128],
                                 hidden_dims_up = [16, 32, 64, 128, 128, 128],
                                 hidden_dims_skip = [0, 0, 0, 0, 0, 0],
                                 filter_size_down = 3,
                                 filter_size_up = 5,
                                 filter_skip_size = 0,
                                 sigmoid_output = self.sigmoid_output,
                                 bias = True,
                                 padding_mode='zero',
                                 upsample_mode='nearest',
                                 downsample_mode='stride',
                                 act_fun='LeakyReLU',
                                 need1x1_up=True
                                )    
    
    def set_input(self, x):
        if type(x) is not list:
            self.input = self.image_check(x)
        else:
            x_c = []
            for x_i in x:
                x_i_c = self.image_check(x_i)
                x_c.append(x_i_c)
            self.input = torch.cat(x_c, 1).float()
            
    def set_target(self, x):

        if type(x) is not list:            
            self.target_channels = x.shape[-1]
            self.target = self.image_check(x)
        else:
            x_c = []
            self.target_channels = []
            for x_i in x:                
                x_i_c = self.image_check(x_i)
                x_c.append(x_i_c)
                self.target_channels.append(x_i.shape[-1])
            self.target = torch.cat(x_c, 1)
       
        if isinstance(self.latent_seed, list) or isinstance(self.latent_seed, tuple):
            self.set_input(self.latent_seed[0]*torch.rand(self.latent_seed[1], *self.target.shape[-2:]))
            
        elif self.latent_seed == 'meshgrid':
            X,Y = torch.meshgrid(torch.arange(0, 1, 1/self.target.shape[-2]),
                                 torch.arange(0, 1, 1/self.target.shape[-1])
                                )
            meshgrid = torch.cat([X[None,:], Y[None,:]])            
            self.set_input(meshgrid)
            
        else:
            self.set_input(self.target) 
    
        self.init_model()   
            
        self.train_dataset = SingleImageDataset(self.target,
                                                input = self.input,
                                                input_noise_std = self.regularization_noise_std,
                                                repetition = self.epoch_steps)

    def set_cluster(self, cluster_labels_2d):
        """
        cluster_labels_2d: np.ndarray or torch.Tensor, shape (H,W), int
        """
        if isinstance(cluster_labels_2d, torch.Tensor):
            t = cluster_labels_2d
        else:
            t = torch.from_numpy(cluster_labels_2d)
        if t.ndim == 2:
            t = t.unsqueeze(0) 
        self.cluster_map = t.long()
        self._cluster_stats = None 
        self._Bmask = None
        self._Win   = None

    def set_mask(self, x):       
        if type(x) is not list:
            flat_mask = torch.tensor(x).view(1,x.shape[-2], x.shape[-1])
            self.mask = torch.stack(self.target_channels*[flat_mask.clone()], 1)
        else:
            masks = []
            for c_idx in range(len(x)):
                x_i = x[c_idx]
                flat_mask = torch.tensor(x_i).view(1, x_i.shape[0], x_i.shape[1])
                deep_mask = torch.stack(self.target_channels[c_idx]*[flat_mask.clone()], 1)
                masks.append(deep_mask)
            self.mask = torch.cat(masks, 1).bool()
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   shuffle=False
                                                  )
        return train_loader
    
    def output(self):
        out = self.forward(self.input.to(self.device)).detach()

        if type(self.target_channels) is list:
            out_list = []
            offset_ch = 0
            for ch in self.target_channels:
                out_list.append(out[0,offset_ch:offset_ch+ch,...].permute(1,2,0).cpu().numpy())
                offset_ch += ch
            return out_list
        else:
            return out[0].permute(1,2,0).cpu().numpy()

    def forward(self, input):
        return self.model.forward(input)
            
    def on_validation_epoch_end(self, outputs):
        if self.logging:
            avg_loss = np.mean([x['val_loss'] for x in outputs])
              
            tensorboard = self.logger.experiment
            a = self.output()

            tensorboard.add_image('Generated',
                                  a,
                                  global_step = self.current_epoch,
                                  dataformats='HWC')

            self.log('val_loss', torch.tensor(avg_loss))      

    def get_loss(self, output, target, mask = None):
        base = F.mse_loss(output, target, reduction='none')
        if mask is not None:
            base = base[mask]
        base_loss = base.mean()
        s2_reg  = torch.tensor(0.0, device=output.device) 
        shape   = torch.tensor(0.0, device=output.device) 

        if (self.cluster_loss_weight > 0.0 and
            isinstance(self.target_channels, list) and len(self.target_channels)>=1 and
            self._cluster_stats is not None and self.cluster_map is not None and self.mask is not None):
            s2_ch = int(self.target_channels[0])
            yhat  = output[:, :s2_ch, ...]      
            clear = self.mask[:, 0, ...]        
            cloud = ~clear
            clusters = self.cluster_map.to(output.device)  
            acc = torch.tensor(0.0, device=output.device); total_pix = 0
            for cid, st in self._cluster_stats.items():
                m = (clusters == cid) & cloud    
                if not m.any(): continue
                low, high = st["low"], st["high"] 
                y = yhat
                below = F.relu(low  - y); above = F.relu(y - high)
                pen = (below**2 + above**2)
                pen = pen[m.expand_as(pen)]
                acc += pen.sum(); total_pix += m.sum().item() * s2_ch
            if total_pix > 0:
                s2_reg = acc / float(total_pix)

        if (self.shape_loss_weight > 0.0 and
            isinstance(self.target_channels, list) and len(self.target_channels)>=1 and
            self._Bmask is not None and self._Win is not None):
            s2_ch = int(self.target_channels[0])
            yhat  = output[:, :s2_ch, ...]        

            if self.mask is not None:
                clear_mask_2d = self.mask[:, 0, ...]            
                cloud_2d = (~clear_mask_2d).float().unsqueeze(1) 
            else:
                cloud_2d = torch.ones_like(self._Win)          

            Win_cloud   = self._Win   * cloud_2d               
            Bmask_cloud = self._Bmask * cloud_2d              

            dx = torch.zeros_like(yhat); dy = torch.zeros_like(yhat)
            dx[:,:,:,1:] = yhat[:,:,:,1:] - yhat[:,:,:,:-1]
            dy[:,:,1:,:] = yhat[:,:,1:,:] - yhat[:,:,:-1,:]

            tv_in = ( (dx**2) * Win_cloud + (dy**2) * Win_cloud ).mean()
            gmag = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12).mean(dim=1, keepdim=True) 
            edge_pen = F.relu(self.edge_tau - gmag) ** 2
            edge_pen = (edge_pen * Bmask_cloud).mean()
            shape = self.tv_in_weight * tv_in + self.edge_align_weight * edge_pen
        total = base_loss + self.cluster_loss_weight * s2_reg + self.shape_loss_weight * shape
        return {'loss': total}

    def on_fit_start(self):
        self.input = self.input.to(self.device)
        self.target = self.target.to(self.device)
        if self.mask is not None:
            self.mask = self.mask.to(self.device)         
        if self.cluster_map is not None:
            self.cluster_map = self.cluster_map.to(self.device)  
        if (self.cluster_loss_weight > 0.0 and
            isinstance(self.target_channels, list) and
            len(self.target_channels) >= 1 and
            self.cluster_map is not None and
            self.mask is not None):
            s2_ch = int(self.target_channels[0])
            s2_tgt = self.target[:, :s2_ch, ...]         
            clear_mask_2d = self.mask[:, 0, ...]         
            clusters = self.cluster_map                    
            uniq = torch.unique(clusters)
            stats = {}
            for cid in uniq.tolist():
                m = (clusters == cid) & clear_mask_2d
                cnt = int(m.sum().item())
                if cnt < self.cluster_min_count:
                    continue 
                m2d  = m.squeeze(0)                                   
                flat = s2_tgt[0].permute(1,2,0).reshape(-1, s2_ch)   
                vals = flat[m2d.reshape(-1)]                        
                low  = torch.quantile(vals, q=self.cluster_q_low,  dim=0).view(1,s2_ch,1,1)
                high = torch.quantile(vals, q=self.cluster_q_high, dim=0).view(1,s2_ch,1,1)
                stats[int(cid)] = {"low": low, "high": high}
            self._cluster_stats = stats

        if (self.shape_loss_weight > 0.0 and self.cluster_map is not None):
            clu = self.cluster_map.to(self.device) 
            H, W = clu.shape[-2:]
            B = torch.zeros((1,1,H,W), device=self.device, dtype=torch.float32)
            diff_x = (clu[:, :, 1:] != clu[:, :, :-1]).float()         
            diff_y = (clu[:, 1:, :] != clu[:, :-1, :]).float()        
            B[:,:,:,1:]  = torch.maximum(B[:,:,:,1:],  diff_x.unsqueeze(1))  
            B[:,:,:,:-1] = torch.maximum(B[:,:,:,:-1], diff_x.unsqueeze(1))  
            B[:,:,1:,:]  = torch.maximum(B[:,:,1:,:],  diff_y.unsqueeze(1))  
            B[:,:,:-1,:] = torch.maximum(B[:,:,:-1,:], diff_y.unsqueeze(1)) 
            if self.edge_dilate > 0:
                k = 2*self.edge_dilate+1
                B = torch.nn.functional.max_pool2d(B, kernel_size=k, stride=1, padding=self.edge_dilate)
            self._Bmask = B.clamp(0,1)               
            self._Win   = (1.0 - self._Bmask)        

    def training_step(self, batch, batch_idx): 
        if self.regularization_noise_std != 0.0:
            input = self.input + self.regularization_noise_std*torch.randn(self.input.shape).to(self.input.device)
        else:
            input = self.input
        out = self.forward(input)
        return self.get_loss(out, self.target, self.mask)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
