import argparse
import os
import time

import cv2
import numpy as np
import torch
from Util.network_util import Build_Generator_From_Dict

device = "cpu"

def get_face(engine, cfg):
    start_time = time.time()
    face = engine([torch.randn(1, 512, device=device)]).detach().cpu().numpy()
    face = np.clip(face[0, ...], -1, 1)
    face = (face + 1) / 2
    print(face.shape)
    face = face.transpose(1, 2, 0)[:, :, ::-1]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return face, elapsed_time

def get_engine(ckpt):
    model_dict = torch.load(ckpt, map_location=device)
    g_ema = Build_Generator_From_Dict(model_dict["g_ema"], size=256).to(
        device
    )
    g_ema.eval()
    return g_ema

def main(cfg):
    engine = get_engine(cfg["dir"])
    face = engine([torch.randn(1, 512, device=device)])

    if cfg["interactive"]:
        wait_time = 1000
        cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)

        while cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) >= 1:
            face, elapsed_time = get_face(engine, cfg)
            face = np.ascontiguousarray(face)
            win_name = f"Elapsed time: {elapsed_time}"
            cv2.putText(
                face,
                win_name,
                (0, int(face.shape[0] * 0.99)),
                cv2.FONT_HERSHEY_SIMPLEX,
                face.shape[0] / 1024,
                (255, 0, 0),
                1 + face.shape[0] // 512,
            )
            cv2.imshow("result", face)

            keyCode = cv2.waitKey(wait_time)
            if (keyCode & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break
    else:
        cv2.imwrite("face.png", face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = vars(args)
    cfg["img_normalize"] = True
    cfg["img_range"] = [-1.0, 1.0]
    cfg["img_rgb2bgr"] = True

    main(cfg)
