# Geometry Auxiliary Salient Object Detection for Light Fields via Graph Neural Networks
![network]()

## Training stage
1. Prepare the training data and change the data path
2. Download the initial model - hqpw to the weights folder
3. Train the model: python3 train_demo.py --mode train


##Testing stage
1. Download the pretrained model- 
2. Download a example of testing data -
3. Test the model: python3 test_demo.py --mode test --sal_mode a --model ./models/geosod_model.pth


Our code is rewritten on the basis of this paper. Thanks a lot for the excellent code provided by the authors.

@inproceedings{zhao2019EGNet, title={EGNet:Edge Guidance Network for Salient Object Detection}, author={Zhao, Jia-Xing and Liu, Jiang-Jiang and Fan, Deng-Ping and Cao, Yang and Yang, Jufeng and Cheng, Ming-Ming}, booktitle={The IEEE International Conference on Computer Vision (ICCV)}, month={Oct}, year={2019}, }