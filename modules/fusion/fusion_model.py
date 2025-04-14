import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.img_fc = nn.Linear(2048, 256)
        self.txt_fc = nn.Linear(768, 128)
        self.lab_fc = nn.Linear(10, 64)
        self.final_fc = nn.Sequential(
            nn.Linear(256 + 128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, txt, lab):
        x1 = torch.relu(self.img_fc(img))
        x2 = torch.relu(self.txt_fc(txt))
        x3 = torch.relu(self.lab_fc(lab))
        x = torch.cat((x1, x2, x3), dim=1)
        return self.final_fc(x)
