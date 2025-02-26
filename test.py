import torch
from model import UNet3D
from data_loader import get_data_loader

def test_model(model_path, data_dir):
    model = UNet3D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.eval().cuda()

    test_loader = get_data_loader(data_dir, batch_size=1, mode="test")

    for batch in test_loader:
        img = batch["image"].cuda()
        with torch.no_grad():
            output = model(img)
        print("Prediction shape:", output.shape)

if __name__ == "__main__":
    test_model("best_model.pth", "data")
