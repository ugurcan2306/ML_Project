import torch
import matplotlib.pyplot as plt

# Load the saved file
data = torch.load('plant_model_complete.pth')
h = data['history']
epochs = range(1, len(h['train_loss']) + 1)

# Create the figure (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Loss (matches Figure 2a/3a in your upload)
ax1.plot(epochs, h['train_loss'], 'b-o', label='Train Loss')
ax1.plot(epochs, h['val_loss'], 'r-o', label='Val Loss')
ax1.set_title('Training vs Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot 2: Accuracy (matches Figure 2b/3b in your upload)
ax2.plot(epochs, h['train_acc'], 'b-o', label='Train Acc')
ax2.plot(epochs, h['val_acc'], 'r-o', label='Val Acc')
ax2.set_title('Training vs Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('my_training_curves.png') # This saves the image for your report!
plt.show()