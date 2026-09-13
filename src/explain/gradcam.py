"""
Grad-CAM explainability.
File location: <project_root>/src/explain/gradcam.py

Why the project needs this
--------------------------
A five-way softmax is not evidence. A model can reach a respectable QWK by
latching onto something irrelevant - lens flare, the black border, a camera
watermark - and the score alone will never tell you. Grad-CAM produces a
heatmap of which pixels drove the prediction, so a human can check the model
looked at haemorrhages and exudates rather than at an artefact.

This is the "explainable" in the project title, and it is the answer to the
examiner's question "how do you know it isn't cheating?".

How it works (Selvaraju et al., 2017)
-------------------------------------
1. Forward pass; keep the activations A of the last convolutional layer.
2. Backward pass from the score for the chosen class; keep the gradients dY/dA.
3. Average each gradient channel spatially -> one importance weight per channel.
   (A channel whose activation strongly raises the class score gets a big weight.)
4. Weighted sum of activation maps, then ReLU - we only want evidence FOR the
   class, not against it.
5. Upsample that coarse map (about 10x10 at 300px input) to the image size.

The map is coarse by construction: it localises regions, not individual
microaneurysms. Say so in your report rather than overclaiming.

Implementation note: activations are captured with a forward hook and
``retain_grad()``, rather than a backward hook. Backward-hook semantics have
changed across PyTorch versions; ``retain_grad`` behaves the same everywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import last_conv_layer


class GradCAM:
    """Grad-CAM for a CNN classifier.

    Usage
    -----
        cam = GradCAM(model)                       # auto-picks the last conv
        heatmap, pred, probs = cam(input_tensor)   # (H, W) float in [0, 1]
        cam.remove_hooks()

    Or as a context manager, which cleans up for you::

        with GradCAM(model) as cam:
            heatmap, pred, probs = cam(x)
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or last_conv_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    # -- hooks ------------------------------------------------------------
    def _forward_hook(self, module, inputs, output):  # noqa: ANN001
        # retain_grad() makes .grad available on this non-leaf tensor after
        # backward, which is what we need for the channel weights.
        output.retain_grad()
        self.activations = output

    def remove_hooks(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove_hooks()

    # -- main -------------------------------------------------------------
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Compute the heatmap for one image.

        Parameters
        ----------
        input_tensor:
            (1, 3, H, W) normalised image, on the same device as the model.
        class_index:
            Which class to explain. ``None`` explains the predicted class - the
            usual choice. Passing the TRUE class for a misclassified image is
            revealing: it shows what the model would have needed to look at.

        Returns
        -------
        (heatmap, predicted_class, probabilities)
            ``heatmap`` is (H, W) float32 in [0, 1] at the input resolution.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[0] != 1:
            raise ValueError(
                f"GradCAM handles one image at a time; got batch of {input_tensor.shape[0]}."
            )

        self.model.zero_grad(set_to_none=True)

        # Gradients are required, so no torch.no_grad() here - and no autocast:
        # fp16 gradients through this can underflow to zero and give a blank map.
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()

        predicted = int(logits.argmax(dim=1).item())
        target = predicted if class_index is None else int(class_index)

        score = logits[0, target]
        score.backward(retain_graph=False)

        if self.activations is None or self.activations.grad is None:
            raise RuntimeError(
                "No gradients captured. The chosen target layer may not be on the "
                "path from input to output. Pass target_layer explicitly."
            )

        activations = self.activations.detach()[0]          # (C, h, w)
        gradients = self.activations.grad.detach()[0]       # (C, h, w)

        # One weight per channel: how much raising this feature map raises the score.
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        cam = (weights * activations).sum(dim=0)            # (h, w)
        cam = F.relu(cam)                                   # keep positive evidence only

        cam = cam[None, None]                               # (1, 1, h, w)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)[0, 0]

        cam = cam.cpu().numpy().astype(np.float32)
        span = float(cam.max() - cam.min())
        # A flat map means no positive evidence anywhere - return zeros rather
        # than dividing by ~0 and amplifying noise into a convincing-looking blob.
        cam = (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

        return cam, predicted, probabilities


# ---------------------------------------------------------------------------
def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a [0,1] heatmap over an RGB uint8 image. Returns RGB uint8.

    Red/yellow = high attention, blue = low.
    """
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"image must be uint8 RGB, got {image_rgb.dtype}")

    if heatmap.shape[:2] != image_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))

    coloured = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    blended = (1 - alpha) * image_rgb.astype(np.float32) + alpha * coloured.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def heatmap_statistics(heatmap: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Simple descriptors of where the attention went.

    ``border_fraction`` is the useful one: if a large share of the high-attention
    area sits in the outer 15% of the frame, the model may be keying on the
    image border rather than on retinal pathology. That is a finding worth
    reporting honestly, not hiding.
    """
    h, w = heatmap.shape
    hot = heatmap >= threshold

    border = np.zeros_like(hot)
    margin_h, margin_w = int(h * 0.15), int(w * 0.15)
    border[:margin_h, :] = border[-margin_h:, :] = True
    border[:, :margin_w] = border[:, -margin_w:] = True

    n_hot = int(hot.sum())
    return {
        "hot_area_fraction": round(float(hot.mean()), 4),
        "border_fraction": round(float((hot & border).sum() / n_hot), 4) if n_hot else 0.0,
        "peak_value": round(float(heatmap.max()), 4),
        "mean_value": round(float(heatmap.mean()), 4),
    }


def cam_for_image(
    model: nn.Module,
    image_rgb: np.ndarray,
    cfg: Dict[str, Any],
    device: torch.device,
    class_index: Optional[int] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: preprocessed RGB image in, everything out.

    Used by the Streamlit app in Milestone 10 - one call gives the prediction,
    the confidence, the full probability vector and the overlay.
    """
    from src.data.transforms import build_transforms

    transform = build_transforms(cfg, train=False)   # never augment at inference
    tensor = transform(image_rgb).unsqueeze(0).to(device)

    with GradCAM(model) as cam:
        heatmap, predicted, probabilities = cam(tensor, class_index=class_index)

    alpha = cfg.get("explain", {}).get("overlay_alpha", 0.4) if alpha is None else alpha

    return {
        "heatmap": heatmap,
        "overlay": overlay_heatmap(image_rgb, heatmap, alpha=alpha),
        "predicted": predicted,
        "confidence": float(probabilities[predicted]),
        "probabilities": probabilities,
        "stats": heatmap_statistics(heatmap),
    }
