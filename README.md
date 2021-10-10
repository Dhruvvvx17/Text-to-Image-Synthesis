# Text-to-Image-Synthesis

This project is part of the course **Topics in Deep Learning** (UE17CS338) taken at PES University during my 6th semester (Spring 2020).

### Introduction about GANs

GANs short for Generative Adversarial Networks are a class of deep learning networks. They consist of two neural networks that compete with each other in order to improve the outcomes of both the networks simultaneously.

GANs are an approach to **Generative Modeling** but rather than the conventional *unsupervised* form of generative modeling that requires a network to discover and learn patterns and regularities in data such that the network can generate outputs which would seem to have been obtained from the original dataspace itself, a GAN converts aims at generating the same through a *supervised* learning process.

A GAN consists of two networks - the **Generator** that generates new output and the **Discriminator** that tries to classify the output as real (from the training data space) or fake (generated). The generator tries to improve by moving from generating absolute noise to generating something close to the real dataset. The discriminator tries to improve by becoming better at differentiating the real output from the fake/generated output. 

Once the generator and discriminator are sufficiently improved and the discriminator is unable to diffrentiate the real from the fake, the training process is complete. The generator can then be used independently to generate output.

<p align="center">
  <img src="https://user-images.githubusercontent.com/35966910/136689516-ee8f5efe-d550-470f-bef0-339b02e18ce8.png" width=600>
</p>

<hr style="border:1px solid gray"> </hr>

### Project Abstract

This project is an implementation of the paper <a href="https://drive.google.com/file/d/1Xah-A_iMofZnLo_RQjLTgG902TXfJmZe/view">Generative Adversarial Text to Image Synthesis</a>. It is a tensorflow based implementation and the text descriptions are encoded using <a href = "https://arxiv.org/abs/1506.06726">Skip Thought Vectors</a>. The below image is the representation of the model architecture.

<p align="center">
  <img src="https://user-images.githubusercontent.com/35966910/136690022-bdb57d4a-d84b-484b-8729-c915033200d6.jpg" width=600>
</p>

### Results

**Training:**

After 120 epochs - ![img_after_120](https://user-images.githubusercontent.com/35966910/136690250-7505a6db-62c8-49b6-8164-d12c150eb1fd.jpg)

After 150 epochs - ![img_after_150](https://user-images.githubusercontent.com/35966910/136690251-1a60ce10-ef6e-4dad-b1eb-591a1b2bcc78.jpg)

After 210 epochs - ![img_after_210](https://user-images.githubusercontent.com/35966910/136690253-ab7db2b7-8350-4e33-b62b-4db514ca8f1b.jpg)

After 240 epochs - ![img_after_240](https://user-images.githubusercontent.com/35966910/136690255-52ef0a77-2f8d-4978-bb2d-d3496f9d762e.jpg)

After 270 epochs - ![img_after_270](https://user-images.githubusercontent.com/35966910/136690248-7920df1e-057e-4dff-8223-75fe6c9e256d.jpg)

After 330 epochs - ![img_after_330](https://user-images.githubusercontent.com/35966910/136690249-e6c000c2-070a-4ad0-9412-838d08ad609a.jpg)

After 450 epochs - ![img_after_450](https://user-images.githubusercontent.com/35966910/136690321-274a6be8-79ad-4979-bb71-3ba9e2f4f9b3.jpg)

After 900 epochs - ![img_after_900](https://user-images.githubusercontent.com/35966910/136690326-d2decd4f-59a5-463c-a900-861db43f4563.jpg)

After 1300 epochs - ![img_after_1300](https://user-images.githubusercontent.com/35966910/136690327-df6e101f-f417-4d5a-bfa4-7efe392e09c1.jpg)

**Testing:**

"A Red Flower" - ![Red_flower_960](https://user-images.githubusercontent.com/35966910/136690402-8fa02109-b88b-4445-85fd-43baec1a6c24.jpg)

"A Blue Flower" - ![Blue_flower_960](https://user-images.githubusercontent.com/35966910/136690404-5982e883-e82c-4b06-bfb8-30edd24252bd.jpg)

"A Purple Flower" - ![Purple_flower_960](https://user-images.githubusercontent.com/35966910/136690405-cc19a029-aa8e-48a6-8c62-693d7db293c1.jpg)

<hr>

**Improvements:**

  - Train for a greater number of epochs to obtain better results
  - Train the model on the MS-COCO data set, and generate more generic images.
  - Try different embedding options for captions(other than skip thought vectors). Also try to train the caption embedding RNN along with the GAN-CLS model.
