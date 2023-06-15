import os
from argparse import ArgumentParser

import numpy as np
import torch

from model.utils import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D
from model.lapirn_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath",
                    default='outputs/cLapIRN/stage/OASIS_CRLoss_one_level_stagelvl3_60000.pth',
                    help="Trained model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='outputs/cLapIRN/reg_out',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='outputs/image_A.nii',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='outputs/image_B.nii',
                    help="moving image")
parser.add_argument("--reg_input", type=float,
                    dest="reg_input", default=0.4,
                    help="Normalized smoothness regularization (within [0,1])")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving
if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = opt.start_channel
reg_input = opt.reg_input


def test():
    print("Current reg_input: ", str(reg_input))

    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow,
                                                                         model_lvl1=model_lvl1).cuda()

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    fixed_img = load_4D(fixed_path)
    moving_img = load_4D(moving_path)

    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():
        reg_code = torch.tensor([reg_input], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)

        F_X_Y = model(moving_img, fixed_img, reg_code)

        X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

        F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

        save_flow(F_X_Y_cpu, savepath + '/contrareg_warpped_flow_' + 'reg' + str(reg_input) + '.nii.gz')
        save_img(X_Y, savepath + '/contrareg_warpped_moving_' + 'reg' + str(reg_input) + '.nii.gz')

    print("Result saved to :", savepath)


if __name__ == '__main__':
    imgshape = (160, 192, 144)
    imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
    imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

    range_flow = 0.4
    test()
