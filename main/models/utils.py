from torch import nn

class TSM(nn.Module):
    def __init__(self, channels, n_segment=20, mode="shift"):
        super().__init__()
        self.n_segment = n_segment
        self.fold = channels // 5
        self.mode = mode 

    def forward(self, x):
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        x = x.view(b, self.n_segment, c, h, w)

        out = x.clone()

        if self.mode == "shift":
            # Original TSM
            out[:, :-1, :self.fold] = x[:, 1:, :self.fold]
            out[:, 1:, self.fold:2*self.fold] = x[:, :-1, self.fold:2*self.fold]

        elif self.mode == "diff":
            # forward diff: next - current
            out[:, :-1, :self.fold] = x[:, 1:, :self.fold] - x[:, :-1, :self.fold]

            # backward diff: current - previous
            out[:, 1:, self.fold:2*self.fold] = x[:, 1:, self.fold:2*self.fold] - x[:, :-1, self.fold:2*self.fold]

        else:
            raise ValueError("mode must be 'shift' or 'diff'")

        return out.view(bt, c, h, w)