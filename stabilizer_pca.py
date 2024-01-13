import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from models.DUT.Smoother import Smoother
from models.DUT.rf_det_so import RFDetSO
from models.DUT.MotionPro import MotionPro

from configs.config import cfg
from utils.MedianFilter import SingleMotionPropagate, MultiMotionPropagate
from utils.image_utils import topk_map
from utils.IterativeSmooth import generateSmooth
import warnings
warnings.filterwarnings("ignore")
from time import sleep
import argparse

device = 'cuda'



class RFDetection(nn.Module):
    def __init__(self, RFDetPath, topK=cfg.TRAIN.TOPK):
        super(RFDetection, self).__init__()

        self.det = RFDetSO(
                cfg.TRAIN.score_com_strength,
                cfg.TRAIN.scale_com_strength,
                cfg.TRAIN.NMS_THRESH,
                cfg.TRAIN.NMS_KSIZE,
                cfg.TRAIN.TOPK,
                cfg.MODEL.GAUSSIAN_KSIZE,
                cfg.MODEL.GAUSSIAN_SIGMA,
                cfg.MODEL.KSIZE,
                cfg.MODEL.padding,
                cfg.MODEL.dilation,
                cfg.MODEL.scale_list,
            )

        self.TOPK = topK

    def forward(self, im_data, batch=2, allInfer=False):
        '''
        @param im_data [B, 1, H, W]
        @return im_topk [B, 1, H, W]
        @return kpts [[N, 4] for B] (B, 0, H, W)
        '''
        if allInfer:
            im_data = im_data
            im_rawsc, _, _ = self.det(im_data)
            im_score = self.det.process(im_rawsc)[0]
            im_topk = topk_map(im_score, self.TOPK).permute(0, 3, 1, 2) # B, 1, H, W
            kpts = im_topk.nonzero()  # (B*topk, 4)
            kpts = [kpts[kpts[:, 0] == idx, :] for idx in range(im_data.shape[0])] # [[N, 4] for B]
            im_topk = im_topk.float()
        else:
            im_topK_ = []
            kpts_ = []
            for j in range(0, im_data.shape[0], batch):
                im_data_clip = im_data[j:j+batch]
                im_rawsc, _, _ = self.det(im_data_clip)
                im_score = self.det.process(im_rawsc)[0]
                im_topk = topk_map(im_score, self.TOPK).permute(0, 3, 1, 2) # B, 1, H, W
                kpts = im_topk.nonzero()  # (B*topk, 4)
                kpts = [kpts[kpts[:, 0] == idx, :] for idx in range(im_data_clip.shape[0])] # [[N, 4] for B]
                im_topk = im_topk.float()
                im_topK_.append(im_topk)
                kpts_ = kpts_ + kpts
            kpts = kpts_
            im_topk = torch.cat(im_topK_, 0)

        return im_topk, kpts # B, 1, H, W; N, 4;

    def reload(self, RFDetPath):

        # self.pwcNet.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(PWCNetPath).items()})
        print('reload RFDet Model')
        pretrained_dict = torch.load(RFDetPath)['state_dict']
        model_dict = self.det.state_dict()
        pretrained_dict = {k[4:]:v for k, v in pretrained_dict.items() if k[:3]=='det' and k[4:] in model_dict}
        assert len(pretrained_dict.keys()) > 0
        model_dict.update(pretrained_dict)
        assert len(model_dict.keys()) == len(pretrained_dict.keys()), 'mismatch for RFDet'
        self.det.load_state_dict(model_dict)
        print("successfully load {} params for RFDet".format(len(model_dict)))
def main():
    parser = argparse.ArgumentParser(description="DUT parameters")
    parser.add_argument('--in_path', required=True, help='Path to input video')
    parser.add_argument('--out_path', required=True, help='Path to output video')
    args = parser.parse_args()
    

    ITER = 40

    print(f'reading frames from : {args.in_path}\n')
    cap = cv2.VideoCapture(args.in_path)
    frames = []
    while True:
        ret,img = cap.read()
        if not ret:
            break
        frames.append(img)
    frames = np.array(frames,dtype=np.uint8)
    num_frames,HEIGHT,WIDTH,_ = frames.shape

    keypointModule = RFDetection('./ckpt/RFDet_640.pth.tar')
    keypointModule.reload('./ckpt/RFDet_640.pth.tar')
    keypointModule.eval().to(device)
    motionpro = MotionPro(HEIGHT,WIDTH,globalchoice='multi')
    state_dict = torch.load('./ckpt/MotionPro.pth')
    motionpro.load_state_dict(state_dict)
    motionpro.to(device).eval()
    smoother = Smoother()
    state_dict = torch.load('./ckpt/smoother.pth')
    smoother.load_state_dict(state_dict)
    smoother.to(device).eval()
    class Grid_Flow(nn.Module):
        def __init__(self):
            super(Grid_Flow, self).__init__()
            self.kpt_detector = keypointModule
            self.raft = models.optical_flow.raft_small(weights = 'Raft_Small_Weights.C_T_V2').eval().to(device)
            self.motionpro = motionpro

        def get_flow(self,img1,img2):
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
            with torch.no_grad():
                flow = self.raft(img1,img2)[-1]
            return flow

        def forward(self,img1,img2):
            img1_gray = torch.mean(img1, dim = 1,keepdim=True)
            im_topk,kpts = self.kpt_detector(img1_gray)
            flow = self.get_flow(img1,img2)
            accurate_flow = flow * im_topk[0,...]
            origin_motion = self.motionpro.inference(accurate_flow[0:1,0:1,...].cuda(),accurate_flow[0:1,1:2,...].cuda(), kpts[0])
            return origin_motion

    #set up motion refinement module
    grid_flow = Grid_Flow()
    grid_flows = torch.zeros((num_frames, 2, HEIGHT // cfg.MODEL.PIXELS, WIDTH // cfg.MODEL.PIXELS)).float()
    for idx in range(num_frames-1):
        print(f'\rExtracting refined flow {idx}',end='')
        torch.cuda.empty_cache()
        img1 = frames[idx,...]
        img2 = frames[idx + 1,...]
        img1 = torch.from_numpy(img1/255.0).unsqueeze(0).permute(0,3,1,2).float().to(device)
        img2 = torch.from_numpy(img2/255.0).unsqueeze(0).permute(0,3,1,2).float().to(device)
        with torch.no_grad():
            refined_grid_motion = grid_flow(img1,img2)
        grid_flows[idx: idx+1,...] = refined_grid_motion
    del frames
    print('\nsmoothing vertex trajectories')
    origin_motion = grid_flows.permute(1,0,2,3).unsqueeze(0)
    del grid_flows
    origin_motion = torch.cat([torch.zeros_like(origin_motion[:, :, 0:1, :, :]).to(origin_motion.device), origin_motion], 2)
    origin_motion = torch.cat([torch.zeros_like(origin_motion[:, :, 0:1, :, :]).to(origin_motion.device), origin_motion], 2)
    origin_motion = torch.cumsum(origin_motion, 2)
    min_value = torch.min(origin_motion)
    origin_motion = origin_motion - min_value
    max_value = torch.max(origin_motion) + 1e-5
    origin_motion = origin_motion / max_value
    with torch.no_grad():
        smoothKernel = smoother(origin_motion.cuda())
    smoothPath = torch.cat(smoother.KernelSmooth(smoothKernel, origin_motion.cuda(), repeat=ITER), 1) # B, 2, T, H, W
    smoothPath = smoothPath * max_value + min_value
    origin_motion = origin_motion * max_value + min_value

    num_components = 5
    U = np.load('./principal_components/PC_U.npy')
    V = np.load('./principal_components/PC_V.npy')
    U1 = np.zeros((num_components, HEIGHT * WIDTH))
    V1 = np.zeros((num_components, HEIGHT * WIDTH))
    for i in range(0, num_components):
        temp = U[i, ...].reshape((256, 512))
        temp = cv2.resize(temp, (WIDTH, HEIGHT))
        U1[i, ...] = temp.reshape(1, HEIGHT * WIDTH)
        temp = V[i, ...].reshape((256, 512))
        temp = cv2.resize(temp, (WIDTH, HEIGHT))
        V1[i, ...] = temp.reshape(1, HEIGHT * WIDTH)
    U = torch.from_numpy(U1).float()
    V = torch.from_numpy(V1).float()
    base = torch.cat((U, V), 0).t()
    base = base.to(device)

    def inpaint(mask, flow, device='cuda'):
        flow = flow.permute(0, 2, 3, 1).to(device)
        mask_flat = mask.view(-1).to(device)
        Q = base[mask_flat > 0].to(device)
        flow_flat = flow.view(-1, 2).to(device)
        valid_flow = flow_flat[mask_flat > 0]
        c = torch.matmul(torch.matmul((torch.matmul(Q.t(), Q) + 0.1 * torch.eye(10).to(device)).inverse(), Q.t()), valid_flow)
        pca_flow = torch.matmul(base, c)
        pca_flow = pca_flow.view(HEIGHT, WIDTH, 2).unsqueeze(0)
        return pca_flow.permute(0,-1,1,2).cpu()

    def show_flow(flow):
        hsv_mask = np.zeros(shape= flow.shape[:-1] +(3,),dtype = np.uint8)
        hsv_mask[...,1] = 255
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=True)
        hsv_mask[:,:,0] = ang /2 
        hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2RGB)
        return(rgb)

    #difference between path with is the sparse warp fields I need
    flows_d = (smoothPath - origin_motion.cuda()).squeeze(0)
    del smoothPath 
    E = flows_d.shape[1]
    flows_d = flows_d.permute(1,0,2,3).contiguous().view(E,2,-1)

    #creating mask and full size flows for pca inpainting
    grid_y, grid_x = torch.meshgrid(torch.arange(0, HEIGHT, cfg.MODEL.PIXELS),
                        torch.arange(0, WIDTH, cfg.MODEL.PIXELS),
                        indexing = 'ij')
    grid_vertices = torch.stack([grid_x, grid_y], dim=-1).view(-1,2)
    mask = torch.zeros((HEIGHT,WIDTH)).float()
    mask[grid_vertices[:,1].long(),grid_vertices[:,0].long()] = 1
    flows_full_sparse  = torch.zeros((E,2,cfg.MODEL.HEIGHT,cfg.MODEL.WIDTH)).float()
    flows_full_sparse[:,:,grid_vertices[:,1],grid_vertices[:,0]] = flows_d.cpu()
    del flows_d
    print('inpainting flows')
    inpainted_flows = torch.zeros_like(flows_full_sparse)
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    for idx in range(inpainted_flows.shape[0]):
        inpainted = inpaint(mask,flows_full_sparse[idx : idx + 1,...])
        inpainted_flows[idx:idx+1,...] = inpainted
        inpainted = inpainted.squeeze(0).permute(1,2,0).numpy()
        cv2.imshow('window',show_flow(inpainted))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyAllWindows()
    del flows_full_sparse
    def warpFlow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return(warped)
    print('warping frames and saving video')
    cap = cv2.VideoCapture(args.in_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_path, fourcc, 30.0, (WIDTH, HEIGHT))
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    idx = 0
    while True:
        ret,img = cap.read()
        if not ret:
            break
        idx += 1
        warped = warpFlow(img, inpainted_flows[idx,...].permute(1,2,0).numpy())
        out.write(warped)
        cv2.imshow('window',warped)
        sleep(1/30)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cv2.destroyAllWindows()
    out.release()
    cap.release()
    
if __name__ == "__main__":
    main()