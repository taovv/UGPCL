PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


python train.py --config='configs/comparison_acdc_224_136/ugpcl_unet_r50.yaml' --device='cuda:1' \
                --work_dir='results/comparison_acdc_224_136'

