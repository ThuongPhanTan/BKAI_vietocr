python3 predict.py \
--device cuda:0 \
--weights ./weights/weight_vietocr_final.pth \
--path_detect  ../../Results/abcnet \
--path_output ../../Results/abcnet_vietocr \
--submission 1

python3 predict.py \
--device cuda:0 \
--weights ./weights/vgg_transformer_AUG.pth \
--path_detect  ../../Results/yolor \
--path_output ../../Results/yolor_vietocr \
--submission 1
