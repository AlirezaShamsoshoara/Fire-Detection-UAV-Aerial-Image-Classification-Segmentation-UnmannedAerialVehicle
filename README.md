# Aerial Imagery dataset for fire detection: classification and segmentation using Unmanned Aerial Vehicle (UAV)

## Paper
<!--- just ou can find the **article** related to this code [here at IEEE](https://ieeexplore.ieee.org/abstract/document/8824917) or --->
You can find the **preprint** from the  ... .
<!--- [Arxiv website](https://arxiv.org/pdf/1904.07380.pdf). --->

### Dataset
* The dataset is uploaded on IEEE dataport. You can find the dataset here at [IEEE Dataport](https://ieee-dataport.org/open-access/aerial-images-pile-fire-detection-using-drones-uavs) or [DOI](https://dx.doi.org/10.21227/qad6-r683).

* This table shows all available data for the dataset. This project uses items 7, 8, 9, and 10 from the dataset. Items 7 and 8 are being used for the "Fire_vs_NoFire" image classification. Items 9 and 10 are for the fire segmentation. 

<!--- ![Alt text](/Output/table.PNG) --->
<img src=/Output/table.PNG width="860" height="600"/>


### Model
* The binary fire classifcation model of this project is based on the Xception Network:

![Alt text](/frames/small_Xception_model.PNG)
<br/>
<br/>

* The fire segmentation model of this project is based on the U-NET:

![Alt text](/frames/u-net-segmentation.PNG)

### Sample
* A short sample video of the dataset is available on YouTube:


## Requirements
* os
* re
* cv2
* copy
* tqdm
* scipy
* numpy
* pickle
* random
* itertools
* Keras 2.4.0
* Tensorflow 2.3.0
* matplotlib.pyplot

## Code
This code is run and tested on Python 3.6 on linux (Ubuntu 18.04) machine with no issues. There is a config file in this directoy which shows all the configuration parameters such as ...

## Results
* Fire classification accuracy:

![Alt text](/Output/classification.PNG)

* Fire classification Confusion Matrix:

<img src=/Output/confusion.PNG width="500" height="500"/>
<!--- ![Alt text](/Output/confusion.PNG) --->

* Fire segmentation metrics and evaluation:

![Alt text](/Output/segmentation.PNG)

* Comparison between generated masks and grount truth mask:

![Alt text](/Output/segmentation_sample.PNG)


## Citation
If you find it useful, please cite our paper as follows:


## License
For academtic and non-commercial usage


