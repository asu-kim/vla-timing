import argparse
import glob
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_PATH = "./models/openvla-7b"
INSTRUCTION = "pick up the nearest object"
UNNORM_KEY  = "bridge_orig"
DEVICE      = "cuda:0"


def fk(joints):
    #Placeholder DH parameters for now, need to measure the actual robot and update these.
    #q1, q2, q3, q4, q5 = joints
    q = joints
    T = T @ np.array([[q, 0, 0, 0],
                      [0, 0, -1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])
    return T


def rot_to_rpy(R):
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def solve_ik(pos, rpy, current_joints):
    #Placeholder for now since the IK isn't working well and I want to focus on the VLA part first. Just return a dummy action that doesn't move the robot.
    joints = 0
    pulses = None
    return joints, pulses


def apply_action(action, current_joints):
    dx, dy, dz, droll, dpitch, _dyaw, gripper = action
    T = fk(current_joints)
    new_pos = T[:3, 3] + np.array([dx, dy, dz]) * 0.05
    new_rpy = rot_to_rpy(T[:3, :3]) + np.array([droll, dpitch, 0.0]) * 0.1
    joints, pulses = solve_ik(new_pos, new_rpy, current_joints)
    if pulses is not None:
        pulses.append(int(np.clip(gripper * 1000, 0, 1000)))
    return joints, pulses


def load_model():
    # Flash attention causes error when installing for some reason, so we use the default implementation for now.
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    return model

'''
def run(image_paths, instruction=INSTRUCTION):
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = load_model()
    #current_joints = np.zeros(5)

    print(f"  Instruction: {instruction}")
    print(f"  Images: {len(image_paths)}")

    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
            print(action)

        # Need to work on the IK and robot control code, so the rest of this is just a placeholder for now.
'''
'''
        new_joints, pulses = apply_action(action, current_joints)
        ik_ok = new_joints is not None

        print(f"[{i+1}/{len(image_paths)}] {path}")
        print(f"action : {np.round(action, 4)}")
        print(f"dx={action[0]:.4f}  dy={action[1]:.4f}  dz={action[2]:.4f}"
              f"droll={action[3]:.4f}  dpitch={action[4]:.4f}  dyaw={action[5]:.4f}"
              f"gripper={action[6]:.4f}")
        if ik_ok:
            print(f"joints: {np.round(np.degrees(new_joints), 1)} deg")
            print(f"pulses: {pulses}")
            current_joints = new_joints
        else:
            print(f"IK:FAILED(target out of reach, holding position)")
        print()

    print(f"  Done. Final joints: {np.round(np.degrees(current_joints), 1)} deg")
'''
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--instruction", default=INSTRUCTION)
    args = parser.parse_args()

    paths = []
    for pattern in args.images:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])

    if not paths:
        print("No images found.")
        raise SystemExit(1)

    run(paths, args.instruction)



if __name__ == "__main__":    
    main()
'''

def start_vla_inference(processor, image, instruction, model):

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = load_model()
    #current_joints = np.zeros(5)

    print(f"  Instruction: {instruction}")
    print(f"  Images: {len(image_paths)}")

    for i, path in enumerate(image_paths):
        image = image.convert("RGB")
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
            print(action)

# Call from the receiver after passing the received image and instruction from the sender. 
#The rest of the code can be adapted from the run() function above, but for now we can just print the action to verify that the VLA inference is working end-to-end.