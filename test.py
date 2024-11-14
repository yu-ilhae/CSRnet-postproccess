import torch
from tqdm import tqdm

from model import CSRNet
from dataset import create_test_dataloader
from config import Config
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, center_of_mass
from skimage.draw import disk
from sklearn.mixture import GaussianMixture

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def denormalize(tensor):
    mean = [0.5, 0.5, 0.5]
    std = [0.225,0.225,0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

cfg = Config(True)                                                          
model = CSRNet().to(cfg.device)                                         
model.load_state_dict(torch.load('./checkpoints/148.pth'))
test_dataloader = create_test_dataloader(cfg.dataset_root)            

model.eval()
total_ae = 0
avg_density = 0

save_path = './save'

os.makedirs(save_path, exist_ok=True)

with torch.no_grad():
    for k, data in enumerate(tqdm(test_dataloader)):
        image = data['image'].to(cfg.device)
        gt_densitymap = data['densitymap'].to(cfg.device)
        et_densitymap = model(image).detach()

        # Calculate GT and ET sums
        gt_sum = gt_densitymap.data.sum().item()
        et_sum = et_densitymap.data.sum().item()
        mae = abs(et_sum - gt_sum)
        
        avg_density += gt_sum
        total_ae += mae

        # Resize images and density maps to 512x512
        image_resized = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
        gt_density_resized = F.interpolate(gt_densitymap, size=(512, 512), mode='bilinear', align_corners=False)
        et_density_resized = F.interpolate(et_densitymap, size=(512, 512), mode='bilinear', align_corners=False)

        image_np = denormalize(image_resized[0].cpu()).permute(1, 2, 0).numpy()
        gt_density_np = gt_density_resized[0, 0].cpu().numpy() 
        et_density_np = et_density_resized[0, 0].cpu().numpy()


        thres1 = 0.001
        thres2 = 0.01
        thres3 = 0.1


        gt_binary1 = (gt_density_np>thres1).astype(int)
        gt_binary2 = (gt_density_np>thres2).astype(int)
        gt_binary3 = (gt_density_np>thres3).astype(int)

        et_binary1 = (et_density_np>thres1).astype(int)
        et_binary2 = (et_density_np>thres2).astype(int)
        et_binary3 = (et_density_np>thres3).astype(int)


        gt_center = []
        bins = [gt_binary1, gt_binary2, gt_binary3]
        cnts = []
        radius = 15
        for i in range(0,3):
            cnt = 0
            binary_map = bins[i]
            labeled_array, num_features = label(binary_map)
            center_map = np.zeros_like(binary_map)
            centers = center_of_mass(binary_map, labeled_array, range(1, num_features + 1))

            for center in centers:
                center_y, center_x = map(int, center) 
                rr, cc = disk((center_y, center_x), radius, shape=center_map.shape)
                center_map[rr, cc] = 1 
                cnt += 1
            gt_center.append(center_map)
            cnts.append(cnt)

        fig, axs = plt.subplots(6, 3, figsize=(15, 30))
        
        axs[0,0].imshow(image_np)
        axs[0,0].axis('off')
        axs[0,0].set_title("Input Image")
        
        axs[0,1].imshow(gt_density_np, cmap='viridis', vmin=0, vmax=0.05)
        axs[0,1].axis('off')
        axs[0,1].set_title(f"GT Density (Sum: {gt_sum:.2f})")

        axs[0,2].imshow(et_density_np, cmap='viridis', vmin=0, vmax=0.05)
        axs[0,2].axis('off')
        axs[0,2].set_title(f"ET Density (Sum: {et_sum:.2f})")




        axs[1,0].imshow(gt_binary1, cmap='grey')
        axs[1,0].axis('off')
        axs[1,0].set_title(f"GT thres: {thres1} Sum: {np.count_nonzero(gt_binary1)})")

        axs[1,1].imshow(gt_binary2, cmap='grey')
        axs[1,1].axis('off')
        axs[1,1].set_title(f"GT thres: {thres2} Sum: {np.count_nonzero(gt_binary2)})")

        axs[1,2].imshow(gt_binary3, cmap='grey')
        axs[1,2].axis('off')
        axs[1,2].set_title(f"GT thres: {thres3} Sum: {np.count_nonzero(gt_binary3)})")



        axs[2,0].imshow(image_np)
        axs[2,0].imshow(gt_center[0], cmap='Greens', alpha=0.5)
        axs[2,0].axis('off')
        axs[2,0].set_title(f"GT CC Sum: {(cnts[0])})")

        axs[2,1].imshow(image_np)
        axs[2,1].imshow(gt_center[1], cmap='Greens', alpha=0.5)
        axs[2,1].axis('off')
        axs[2,1].set_title(f"GT CC Sum: {(cnts[1])})")

        axs[2,2].imshow(image_np)
        axs[2,2].imshow(gt_center[2], cmap='Greens', alpha=0.5)
        axs[2,2].axis('off')
        axs[2,2].set_title(f"GT CC Sum: {(cnts[2])})")




        axs[3,0].imshow(et_binary1, cmap='grey')
        axs[3,0].axis('off')
        axs[3,0].set_title(f"ET thres: {thres1} Sum: {np.count_nonzero(et_binary1)})")

        axs[3,1].imshow(et_binary2, cmap='grey')
        axs[3,1].axis('off')
        axs[3,1].set_title(f"ET thres: {thres2} Sum: {np.count_nonzero(et_binary2)})")

        axs[3,2].imshow(et_binary3, cmap='grey')
        axs[3,2].axis('off')
        axs[3,2].set_title(f"ET thres: {thres3} Sum: {np.count_nonzero(et_binary3)})")


        et_center = []
        bins = [et_binary1, et_binary2, et_binary3]
        cnts = []
        radius = 15
        for i in range(0,3):
            cnt = 0
            binary_map = bins[i]
            labeled_array, num_features = label(binary_map)
            center_map = np.zeros_like(binary_map)
            centers = center_of_mass(binary_map, labeled_array, range(1, num_features + 1))

            for center in centers:
                center_y, center_x = map(int, center) 
                rr, cc = disk((center_y, center_x), radius, shape=center_map.shape)  
                center_map[rr, cc] = 1  
                cnt += 1
            et_center.append(center_map)
            cnts.append(cnt)


        axs[4,0].imshow(image_np)
        axs[4,0].imshow(et_center[0], cmap='Greens', alpha=0.5)
        axs[4,0].axis('off')
        axs[4,0].set_title(f"ET CC Sum: {(cnts[0])})")

        axs[4,1].imshow(image_np)
        axs[4,1].imshow(et_center[1], cmap='Greens', alpha=0.5)
        axs[4,1].axis('off')
        axs[4,1].set_title(f"ET CC Sum: {(cnts[1])})")

        axs[4,2].imshow(image_np)
        axs[4,2].imshow(et_center[2], cmap='Greens', alpha=0.5)
        axs[4,2].axis('off')
        axs[4,2].set_title(f"ET CC Sum: {(cnts[2])})")

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'{k}.png'))
        plt.close(fig)


print('Total number of validation dataset:', len(test_dataloader))
print('Average density:', avg_density / len(test_dataloader))
print('MAE:', total_ae / len(test_dataloader))
