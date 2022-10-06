import argparse

import torch
import torch.nn as nn
from Util.network_util import Build_Generator_From_Dict

device = "cpu"
gpu_device_ids = [0, 1]

# Arg Parsing

parser = argparse.ArgumentParser()

parser.add_argument("--generated_img_size", type=int, default=256)
parser.add_argument(
    "--ckpt", type=str, default="""./Model/full_size_model/256px_full_size.pt"""
)

args = parser.parse_args()

model_dict = torch.load(args.ckpt, map_location=device)
g_ema = Build_Generator_From_Dict(model_dict["g_ema"], size=args.generated_img_size).to(
    device
)
# g_ema = nn.DataParallel(g_ema, device_ids=gpu_device_ids)
g_ema.eval()

latent = torch.randn(1, 512, device=device)

torch.onnx.export(
    g_ema,
    [latent],
    "cagan.onnx",
    input_names=["var"],
    output_names=["img"],
    verbose=True,
    opset_version=11,
)
