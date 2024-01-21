import torch
import warnings
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CropSegmentationDataset
from model3 import DeepLabV3Plus
from losses import PixelWiseCrossEntropy
from trainer2 import BaselineTrainer


warnings.simplefilter("ignore")

def main():

    # Define any additional transformations if needed
    train_input_transform = transforms.Compose([

        transforms.Resize((572, 572)),
        transforms.RandomResizedCrop(120),  # Adjust the crop size as needed

        transforms.ToTensor(),
    ])
    val_input_transform = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor(),
    ])

    train_target_transform = transforms.Compose([
        transforms.Resize((388, 388)),
        transforms.RandomResizedCrop(120),  # Adjust the crop size as needed

        transforms.ToTensor(),
    ])
    val_target_transform = transforms.Compose([
        transforms.Resize((388, 388)),
        transforms.RandomResizedCrop(120),  # Adjust the crop size as needed

        transforms.ToTensor(),
    ])



    # Instantiate dataset
    train_dataset = CropSegmentationDataset(transform=train_input_transform,target_transform=train_target_transform,
                                            set_type="train")
    val_dataset = CropSegmentationDataset(transform=val_input_transform,target_transform=val_target_transform,set_type="val")


    # Create DataLoader for train and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,)

    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create model instance
    model = DeepLabV3Plus(num_classes=5)  # Adjust num_classes based on your dataset

    loss = PixelWiseCrossEntropy()

    optimizer = torch.optim.Adam(model.parameters())

    trainer = BaselineTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        use_cuda=False
    )

    train_loss = trainer.fit(train_data_loader=train_dataloader, val_data_loader=val_dataloader, epoch=2)

    trainer.save_results(val_data_loader=val_dataloader)

    print(f"Training loss is: {train_loss},")




if __name__ == "__main__":
    main()