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

</br>

### Project Abstract


