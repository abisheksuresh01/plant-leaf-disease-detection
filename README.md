# Plant Leaf Disease Detection Using CNN

This project leverages Convolutional Neural Networks (CNNs) to detect diseases in plant leaves using the PlantVillage dataset. The process involves data preparation, model architecture design, training, and evaluation.

## Project Structure

- **dataset_generator.py**: Script for generating lists of healthy and unhealthy images from the PlantVillage dataset.
- **train.py**: Main training script for the CNN models.
- **cifar10.py**: Reference script containing CIFAR-10 dataset loading and model architecture examples.
- **healthy.txt**: List of paths to healthy leaf images.
- **unhealthy.txt**: List of paths to unhealthy leaf images.
- **resnet.png**: Image of the ResNet model architecture.
- **result_plant.png**: Example result image of the plant disease detection.

## Data Preparation

The dataset is segmented into healthy and unhealthy leaf images, normalized, and resized to 32x32 pixels.

## Model Architectures

Three CNN architectures are explored:
1. **Model 1**: A simple sequential CNN.
2. **Model 2**: A CNN with residual blocks.
3. **Model 3**: A complex CNN with dropout and batch normalization.

## Training

The models are trained using early stopping, learning rate reduction, and model checkpointing. The training script (`train.py`) handles data loading, normalization, and model training.

## Evaluation

The trained models are evaluated based on accuracy and loss on the testing dataset. Performance metrics and visualizations are provided to assess the models' effectiveness.

## Results

The project includes visualizations of the training and validation loss and accuracy over epochs to understand the model's learning behavior.

## Usage

1. Generate the dataset lists:
   ```bash
   python dataset_generator.py
