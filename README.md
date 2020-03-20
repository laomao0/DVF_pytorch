re-implementation of DVF

origin link: https://github.com/lxx1991/pytorch-voxel-flow

we re-implement the sync_bn using cupy >= 5.0.0, and provide a test script for frame prediction
the frame interpolation is not test

running envs: cupy >= 5.0.0, pytorch >= 1.0.0

test prediction
1. download the UCF101-test: https://drive.google.com/file/d/1EDIn8gxHpApmVPzDts2zc45ixbnqAegp/view
2. modify dataset path of core/datasets/ucf_101_test_new.py
     for example:  dataset_path = '/DATA/wangshen_data/UCF101/ucf101_extrap_ours'
3. download the pre-trained weight: https://drive.google.com/file/d/1FB-mpS4UokiLriDBNJSBmozMQRH0Qez1/view
4. modify model path of test_prediction.py
     for example: checkpoint_path = '/DATA/wangshen_data/UCF101/voxelflow_finetune_model_best.pth.tar'
5. run the prediction process:
     python test_prediction.py configs/voxel-flow.py



# runing results:
  Testing Results: PSNR 31.457 (31.4573)  Loss 0.00856
  31.457312964287905