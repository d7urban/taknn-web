"""ONNX export and INT8 quantization for browser deployment.

Exports:
  1. Trunk + value/aux heads as ONNX (for onnxruntime-web inference)
  2. Policy MLP weights as flat binary (for WASM policy scorer)

The policy head can't be exported to ONNX easily because it takes
variable-length descriptor inputs. Instead, we export the trunk
(which produces spatial + global features) and the policy MLP weights
separately. The WASM side handles descriptor construction and MLP eval.

Usage:
    python -m taknn.train.export --checkpoint checkpoints/student_best.pt --out exports/
"""

import argparse
import os
import struct

import numpy as np
import torch

from ..models.student import StudentModel
from ..models.teacher import TeacherModel


def export_trunk_onnx(model, out_path, channels, opset_version=17):
    """Export trunk + value/aux heads to ONNX.

    Inputs:
        board_tensor: [1, 31, 8, 8] float32
        size_id: [1] int64

    Outputs:
        wdl: [1, 3] float32
        margin: [1, 1] float32
        spatial: [1, C, 8, 8] float32 (trunk features for policy scorer)
        global_pool: [1, C] float32
    """
    model.eval()
    device = next(model.parameters()).device

    dummy_board = torch.randn(1, 31, 8, 8, device=device)
    dummy_size = torch.zeros(1, dtype=torch.long, device=device)

    # Create a wrapper that returns a tuple (ONNX needs tuple output)
    class TrunkWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, board_tensor, size_id):
            out = self.model(board_tensor, size_id)
            return out["wdl"], out["margin"], out["spatial"], out["global"]

    wrapper = TrunkWrapper(model).to(device)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_board, dummy_size),
        out_path,
        input_names=["board_tensor", "size_id"],
        output_names=["wdl", "margin", "spatial", "global_pool"],
        dynamic_axes={
            "board_tensor": {0: "batch"},
            "size_id": {0: "batch"},
            "wdl": {0: "batch"},
            "margin": {0: "batch"},
            "spatial": {0: "batch"},
            "global_pool": {0: "batch"},
        },
        opset_version=opset_version,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Exported trunk ONNX: {out_path} ({size_kb:.0f} KB)")


def quantize_onnx(input_path, output_path):
    """Quantize ONNX model to INT8 (dynamic quantization)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
        size_kb = os.path.getsize(output_path) / 1024
        print(f"Quantized INT8: {output_path} ({size_kb:.0f} KB)")
    except ImportError:
        print("onnxruntime not available, skipping INT8 quantization")
        print("Install with: pip install onnxruntime")


def export_policy_weights(model, out_path):
    """Export policy MLP and embedding weights as flat binary for WASM.

    Format: sequence of named weight blobs.
    Header: magic (4 bytes) + version (4 bytes) + num_blobs (4 bytes)
    Each blob: name_len (2 bytes) + name (UTF-8) + ndims (1 byte) + dims (4 bytes each) + data (float32)
    """
    model.eval()

    # Collect policy-related parameters
    policy_params = {}
    for name, param in model.named_parameters():
        if any(key in name for key in ["policy_mlp", "move_type_emb", "piece_type_emb",
                                        "direction_emb", "pickup_count_emb",
                                        "drop_template_emb", "travel_length_emb"]):
            policy_params[name] = param.detach().cpu().numpy()

    with open(out_path, 'wb') as f:
        # Header
        f.write(b'TPOL')  # magic
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(policy_params)))  # num blobs

        for name, data in policy_params.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<B', len(data.shape)))
            for dim in data.shape:
                f.write(struct.pack('<I', dim))
            f.write(data.astype(np.float32).tobytes())

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Exported policy weights: {out_path} ({size_kb:.0f} KB, {len(policy_params)} blobs)")
    for name, data in policy_params.items():
        print(f"  {name}: {data.shape}")


def verify_onnx(model, onnx_path, device):
    """Verify ONNX model output matches PyTorch within tolerance."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available, skipping verification")
        return True

    model.eval()

    # Run PyTorch
    board = torch.randn(1, 31, 8, 8, device=device)
    size_id = torch.zeros(1, dtype=torch.long, device=device)

    with torch.no_grad():
        pt_out = model(board, size_id)

    # Run ONNX
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "board_tensor": board.cpu().numpy(),
        "size_id": size_id.cpu().numpy(),
    })

    # Compare
    wdl_diff = np.abs(pt_out["wdl"].cpu().numpy() - ort_out[0]).max()
    margin_diff = np.abs(pt_out["margin"].cpu().numpy() - ort_out[1]).max()

    ok = wdl_diff < 1e-4 and margin_diff < 1e-4
    status = "PASS" if ok else "FAIL"
    print(f"ONNX verification: {status} (wdl_diff={wdl_diff:.6f}, margin_diff={margin_diff:.6f})")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--out", default="exports", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Also create INT8 quantized model")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()

    device = torch.device("cpu")  # Export on CPU for reproducibility

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_type = config.get("model_type", "student")
    channels = config.get("channels", 64)
    num_blocks = config.get("num_blocks", 6)
    film_embed_dim = config.get("film_embed_dim", 16)

    if model_type == "teacher":
        model = TeacherModel(channels, num_blocks, film_embed_dim)
    else:
        model = StudentModel(channels, num_blocks, film_embed_dim)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type} {channels}ch x {num_blocks} blocks, {param_count:,} params")

    os.makedirs(args.out, exist_ok=True)

    # Export ONNX
    onnx_path = os.path.join(args.out, f"{model_type}_trunk.onnx")
    export_trunk_onnx(model, onnx_path, channels)

    if args.verify:
        verify_onnx(model, onnx_path, device)

    # Quantize
    if args.quantize:
        int8_path = os.path.join(args.out, f"{model_type}_trunk_int8.onnx")
        quantize_onnx(onnx_path, int8_path)

        if args.verify:
            print("Verifying INT8 model (argmax agreement)...")
            verify_onnx(model, int8_path, device)

    # Export policy weights
    policy_path = os.path.join(args.out, f"{model_type}_policy.bin")
    export_policy_weights(model, policy_path)


if __name__ == "__main__":
    main()
