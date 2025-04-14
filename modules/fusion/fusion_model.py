import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.image_branch = nn.Linear(2048, 512)
        self.text_branch = nn.Linear(768, 256)
        self.lab_branch = nn.Linear(10, 64)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_txt, x_lab):
        x1 = self.image_branch(x_img)
        x2 = self.text_branch(x_txt)
        x3 = self.lab_branch(x_lab)
        x = torch.cat((x1, x2, x3), dim=1)
        return self.classifier(x)
