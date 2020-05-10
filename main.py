# Driver file, imports all other files and runs an end to end flow of the project
# 1. Downloads datasets
# 2. Data Loader
# 3. Training
# 4. Generating thought vectors for given captions.
# 5. Generating sample images

import download_dataset
import data_loader
import train
import generate_thought_vectors
import generate_images

if __name__ == "__main__":
    
    print("Starting dataset download")
    download_dataset.startDownload()
    print("Download Complete!\n")
    
    print("Starting data loader operation.\nThis may take around 6hrs as dataset contains 8000 images.")
    data_loader.startEmbeddings()
    print("Embeddings Complete!\n")
    
    print("Starting training,\nThis may take around 8hrs for 10epochs, each having 96 batches.")
    train.startTraining()
    print("Training Complete!\n")

    print("Creating thought vectors,\nFor given caption file.")
    generate_thought_vectors.createThoughtVectors()
    print("Thought vectors created!\n")

    print("Generating image for given caption.")
    generate_images.createImages()
    print("Image generated.\n")

    print("Check the val_sample sub directory in the data directory for the generated images.")