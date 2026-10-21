# LEGACY / UNVALIDATED PROTOTYPE: attribution correctness is not established.
# Source preservation: command-line execution resolves repository-relative inputs.
if __name__ == '__main__':
    import os as _preserve_os, sys as _preserve_sys
    from pathlib import Path as _PreservePath
    _preserve_root = _PreservePath(__file__).resolve().parents[0]
    _preserve_os.chdir(_preserve_root)
    _preserve_sys.path.insert(0, str(_preserve_root))
    _preserve_sys.path.insert(0, str(_preserve_root / 'src' / 'data'))

import torch
import torch.nn.functional as F
from torchvision.models.video import swin3d_t
from torchvision.models.video.swin_transformer import (
    _compute_pad_size_3d,
    _compute_attention_mask_3d,
    _get_window_and_shift_size,
)


saved = {}


def capture_gradient_attention(module, input, output):
    """
    Recompute the attention matrix of the LAST VideoSwin attention block
    from the block input, and keep its gradient.
    """
    x = input[0]

    b, t, h, w, c = x.shape

    window_size = module.window_size.copy()
    shift_size = module.shift_size.copy()

    window_size, shift_size = _get_window_and_shift_size(
        shift_size,
        [t, h, w],
        window_size,
    )

    relative_position_bias = module.get_relative_position_bias(window_size)

    pad_size = _compute_pad_size_3d(
        (t, h, w),
        tuple(window_size),
    )

    x = F.pad(
        x,
        (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]),
    )

    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)

    # Shifted window
    if sum(shift_size) > 0:
        x = torch.roll(
            x,
            shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
            dims=(1, 2, 3),
        )

    num_windows = (
        (tp // window_size[0])
        * (hp // window_size[1])
        * (wp // window_size[2])
    )

    x = x.view(
        b,
        tp // window_size[0],
        window_size[0],
        hp // window_size[1],
        window_size[1],
        wp // window_size[2],
        window_size[2],
        c,
    )

    x = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(
            b * num_windows,
            window_size[0] * window_size[1] * window_size[2],
            c,
        )
    )

    # Q, K, V
    qkv = F.linear(x, module.qkv.weight, module.qkv.bias)

    qkv = qkv.reshape(
        x.size(0),
        x.size(1),
        3,
        module.num_heads,
        c // module.num_heads,
    ).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * (c // module.num_heads) ** -0.5

    # Attention logits
    attn = q.matmul(k.transpose(-2, -1))
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        attn_mask = _compute_attention_mask_3d(
            x,
            padded_size,
            tuple(window_size),
            tuple(shift_size),
        )

        attn = attn.view(
            x.size(0) // num_windows,
            num_windows,
            module.num_heads,
            x.size(1),
            x.size(1),
        )

        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)

        attn = attn.view(
            -1,
            module.num_heads,
            x.size(1),
            x.size(1),
        )

    attn = F.softmax(attn, dim=-1)

    saved["attention"] = attn

    print("Captured attention shape:", attn.shape)
    print("requires_grad:", attn.requires_grad)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = swin3d_t(weights=None)

    # Binary head
    model.head = torch.nn.Linear(
        model.head.in_features,
        2,
    )

    model = model.to(device)
    model.eval()

    target_attention = model.features[6][1].attn

    handle = target_attention.register_forward_hook(
        capture_gradient_attention
    )

    # Fake 16-frame video
    x = torch.randn(
        1,
        3,
        16,
        224,
        224,
        device=device,
        requires_grad=True,
    )

    logits = model(x)

    print("Logits:", logits)

    score = logits[:, 0].sum()

    score.backward()

    attn = saved["attention"]

    print("Attention gradient:", attn.grad)

    handle.remove()


if __name__ == "__main__":
    main()