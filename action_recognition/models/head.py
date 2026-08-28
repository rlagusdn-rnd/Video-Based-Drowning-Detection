import torch.nn as nn

class VideoMAEv2Head_Linear(nn.Module):
    def __init__(self, in_dim=768, num_classes=2, head_drop_rate=0.0):
        super().__init__() # layer 초기화 (필수)
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(in_dim, num_classes)
        
    def forward(self, x):
        x = self.head_dropout(x)
        return self.head(x)
    
class VideoMAEv2Head(nn.Module):
    def __init__(self, in_dim=768, num_classes=2, head_drop_rate=0.0):
        super().__init__() # layer 초기화 (필수)
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(in_dim, num_classes)
        
    def forward(self, x):
        x = self.head_dropout(x)
        return self.head(x)