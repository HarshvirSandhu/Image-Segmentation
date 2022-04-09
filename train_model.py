from Unet import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from seg_dataset import load_data
import torch
import torch.nn as nn
import torch.optim as optim

LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 3
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
img_path = "D:/segmentation dataset/supervisely_person_clean_2667_img/images"
mask_path = "D:/segmentation dataset/supervisely_person_clean_2667_img/masks"

train_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    # A.ToGray(always_apply=True, p=1),
    # A.Rotate(limit=25, p=0.4),
    # A.VerticalFlip(p=0.1),
    # A.HorizontalFlip(p=0.3),
    ToTensorV2()
])

train_loader = load_data(img_path, mask_path, BATCH_SIZE, transform=train_transform)

model = UNET()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    print("Epoch:", epoch)
    epoch_loss = 0
    for i, (data, targets) in enumerate(train_loader):
        targets = targets.unsqueeze(1)
        print(targets.shape)
        print(f"{i}/{len(train_loader)}")
        pred = model(data)
        loss = criterion(pred, targets)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss at epoch {epoch}: {epoch_loss}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, "Unetmodelcheckpoint.pth.tar")


checkpoint = torch.load("Unetmodelcheckpoint.pth.tar", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

# Testing the model on a single image

import cv2
from torchvision.utils import save_image
# img = torch.rand((1, 3, 100, 100))
# a = model(img)
# print(a.shape)
img_path = "B.jpg"
img = cv2.imread(img_path)
i = train_transform(image=img)
# print(i['image'].unsqueeze(0))

a = model(i['image'].unsqueeze(0))
# print(i.shape)
save_image(a, "check.jpg")
image = cv2.imread("check.jpg")
img = cv2.resize(img, (500, 500))
a = cv2.bitwise_and(img, image)
cv2.imshow("look", a)
cv2.imshow("2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()