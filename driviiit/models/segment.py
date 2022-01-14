import numpy as np
import torch, torch.utils.data
import cv2 as cv
import segmentation_models_pytorch as smp

from driviiit.interface.config import DEVICE
from driviiit.interface.config import SEGMENTATION_COLORS_MAP


class LiveSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images = torch.from_numpy(images.transpose(0, 3, 1, 2)) / 255 - 0.5
        self.masks = torch.from_numpy(masks)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        return image, mask

    def __len__(self):
        return len(self.images)

    def loader(self, batch_size=2):
        return torch.utils.data.DataLoader(self, batch_size=batch_size)


class LiveSegmentationTrainer:
    def __init__(self, load=False):
        self.encoder_name = "resnet18"
        self.model = (
            smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights="imagenet",
                classes=1,
                activation="sigmoid",
            )
            if not load
            else torch.load(f"./data/models/segmentation_unet_{self.encoder_name}.pth")
        )
        self.loss = smp.utils.losses.DiceLoss()
        self.optimizer = torch.optim.Adam(
            [dict(params=self.model.parameters(), lr=0.0001)]
        )
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

    def train(self, training_dataset: LiveSegmentationDataset, num_epochs=5):
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=DEVICE,
            verbose=True,
        )
        train_loader = training_dataset.loader()
        for i in range(1, num_epochs + 1):
            _train_logs = train_epoch.run(train_loader)
        torch.save(
            self.model, f"./data/models/segmentation_unet_{self.encoder_name}.pth"
        )

    def predict(self, prediction_dataset: LiveSegmentationDataset):
        predict_loader = prediction_dataset.loader()
        predictions = []
        for batch in predict_loader:
            predictions.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(predictions)


def train(trainer, filename):
    data = np.load(f"data/records/{filename}.npz")
    road_masks = np.all(
        np.equal(data["segm_front"], SEGMENTATION_COLORS_MAP["ROAD"]), axis=3
    )
    dataset = LiveSegmentationDataset(data["camera_front"], road_masks)
    trainer.train(training_dataset=dataset)


def visualize(trainer, filename):
    data = np.load(f"data/records/{filename}.npz")
    for image in data["camera_front"]:
        if cv.waitKey(100) == ord("q"):
            break
        tensor = (
            torch.from_numpy(image.transpose(2, 0, 1) / 255.0)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )
        mask = trainer.model(tensor).squeeze().detach().cpu().numpy()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow("image", image)
        cv.imshow("mask", mask)
    cv.destroyAllWindows()


if __name__ == "__main__":
    agent = LiveSegmentationTrainer(load=True)
    for i in range(5):
        train(agent, f"trajectory_0001.{i+1:03}")
    # visualize(agent, "trajectory_0001.000")
