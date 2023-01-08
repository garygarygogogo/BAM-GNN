import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.mpnn import MPNN
from model.mpnn import MPNN2
from model.mpnn import MPNN3
from model.ASPP import ASPP
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from util.util import get_train_val_set


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        
        print(args.ppm_scales)
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.ppm_scales = args.ppm_scales
        self.mpnn = MPNN2(256, args.no_pass, args.device_number)
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq/2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        
        assert self.layers in [50, 101, 152]
    
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet'+str(args.layers)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)               
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try: 
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512           
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.pyramid_bins = self.ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.compressor = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.5),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(1, 1), padding=0, bias=True),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.5))
        
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.res2 = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(reduce_dim, reduce_dim, kernel_size=(3, 3), padding=1, bias=True),
                                  nn.ReLU(inplace=True))

        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.beta_conv.parameters()},
                {'params': model.inner_cls.parameters()},
                {'params': model.res1.parameters()},
                {'params': model.res2.parameters()},
                {'params': model.mpnn.parameters()},
                {'params': model.compressor.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},
                {'params': model.kshot_rw.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [     
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.beta_conv.parameters()},
                {'params': model.inner_cls.parameters()},
                {'params': model.res1.parameters()},
                {'params': model.res2.parameters()},
                {'params': model.mpnn.parameters()},
                {'params': model.compressor.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},               
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False


    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None):
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        # Support Feature
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = [] 
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat, mask)
            supp_pro_list.append(supp_feat)
            supp_feat_list.append(eval('supp_feat_' + self.low_fea_id))

        # K-Shot Reweighting
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1,2))/norm_max).reshape(bs,1,1,1)) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
            
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
            
            supp_feat = supp_pro_list[0]
            for i in range(1, len(supp_pro_list)):
                supp_feat += supp_pro_list[i]
            supp_feat /= len(supp_pro_list)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1,True) # [bs, 1, 1, 1]            

        # Prior Similarity Mask
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1,True)

        out_list = []
        pyramid_feat_list = []

        # Support feature list o anki indekse ait tüm batchlerdeki embeddingleri tutar.
        # B,C,1,1 - 0shot B,C,1,1 - 1shot ... tüm hepsinin ortalamasıda her bir episode için masked global average
        # pooled supportların ortalamasını veriyor. Sonuç olarak B,C,1,1 ortalama maskeleri tutuyor.
        # PRIOR MASKLERIN boyutunun da B,1,60,60 olduğunu düşünüyorum.
        ## SAGNN Buradan itibaren ekleyebilirsin.
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)  # !
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)
            merge_feat_bin = self.beta_conv[idx](merge_feat_bin)
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        # H = torch.stack(pyramid_feat_list, 1)
        H = self.mpnn(pyramid_feat_list)
        H = self.compressor(torch.cat(H, 1))
        H = self.res1(H) + H
        H = self.res2(H) + H

        # Base and Meta
        base_out = self.learner_base(query_feat_4)

        query_meta = self.ASPP_meta(H)
        query_meta = self.res1_meta(query_meta)   # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta 
        meta_out = self.cls_meta(query_meta)
        
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:,0:1,:,:]                           # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:,1:,:,:]                            # [bs, 1, 60, 60]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes+1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)
                base_map_list.append(base_out_soft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
            base_map = torch.cat(base_map_list,0)
            # <alternative implementation>
            # gather_id = (cat_idx[0]+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
            # fg_map = base_out_soft.gather(1,gather_id)
            # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map            
        else:
            base_map = base_out_soft[:,1:,:,:].sum(1,True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            # Added: GNN Loss
            aux_loss1 = torch.zeros_like(main_loss)

            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss1 = aux_loss1 + self.criterion(inner_out, y_m.long())
            aux_loss1 = aux_loss1 / len(out_list)
            aux_loss2 = self.criterion(base_out, y_b.long())
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return final_out, meta_out, base_out

