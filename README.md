# Source code for the paper "Testing the Channels of Convolutional Neural Networks"
FtGAN is an extension of GAN designed to test the channels of CNNs, aiding in the understanding and debugging of these networks. 
FtGAN can generate test data with varying the intensity (i.e., sum of the neurons) of a channel of a target CNN. 

We share the source files for running the experiments in the paper. 
Particularly, we include the automated scripts for running the experiments in Section 7.2, but the code can be
simply modified to run all other experiments.

This code is to run FtGAN to test the defective AlexNet model trained 
with the MNIST_Swell Dataset in Section 7.2.

## 1. Requirements

CUDA version 10.0 and python3.7 are required to run the demo.

To install other requirements (Python Libraries) execute the following command:

```setup
pip install -r requirements.txt
```


## 2. Downloading Our Trained FtGAN Instance

Because it may take a few hours to pre-train (and fine-tune) an FtGAN instance, 
we made our FtGAN instance in the paper downloadable. <br>
To download our model, click the following link and unzip the compressed file in the current directory.

- [FtGAN Instance](https://drive.google.com/file/d/18V_Es0Vy74Go3mGWA-TxDIMGfsj4qtH1/view?usp=sharing)



## 3. Test Data Generation with FtGAN


To use the downloaded FtGAN instance to generate test data, run the following command:

```test
sh genTest_MNIST_SWELLON.sh
```

The script will generate test images that varies the intensity of the channel 234 in the 4th layer of the (defective) AlexNet instance; <br>
the channel intensity is 0.5x -- 2.5x of that of the original images (seed).
The generated test images are stored in output/FtGAN_mnist_l4_ch234/sample_test_data/test[0-99].png (100 images).<br>
Each png file contains an original image and five test images generated from the original image with 
gradually changing the intensity of the tested channel from 0.5x -- 2.5x of the original intensity.


## 4. Training an FtGAN Instance

To train a new FtGAN instance (e.g. for channel 5 in layer 3), you can use the following command:

```train
sh trainFTGAN_MNIST.sh -l 3 -c 5
```   
 
Then FtGAN is trained to generate test data that varies the intensity of the selected channel (channel 5 in layer 3). <br>
The training includes both pre-training and fine-tuning; the training takes about 10 minutes on Titan XP.




