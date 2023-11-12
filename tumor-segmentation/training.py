from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim

# Custom Dataset Class
class MIPDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return {'image': image, 'mask': mask}

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Assuming you have lists of file paths: train_image_paths, train_mask_paths, val_image_paths, val_mask_paths
train_dataset = MIPDataset(train_image_paths, train_mask_paths, transform=transform)
val_dataset = MIPDataset(val_image_paths, val_mask_paths, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['image'].to(device))
        loss = criterion(outputs, batch['mask'].to(device))
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            outputs = model(batch['image'].to(device))
            loss = criterion(outputs, batch['mask'].to(device))
            val_loss += loss.item()
        val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

# You can also use the healthy control images as additional negative samples to ensure the model
# is not over-identifying tumorous areas. This can be done by including them in the training loop
# with corresponding zero masks.
