import torch
import argparse
from loguru import logger
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.data import get_yolox_datadir
import os
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead  # Ensure these are correctly imported
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def _get_dataset(input_size, cache: bool, json_file: str = "instances_train2017.json", name: str = "train2017", cache_type: str = "ram", flip_prob:float=.5, hsv_prob:float=1):
    from yolox.data import COCODataset, TrainTransform
    return COCODataset(
        data_dir=os.path.join(get_yolox_datadir(), "COCO"),
        img_size=input_size,
        cache=cache,
        json_file=json_file,
        name=name,
        cache_type=cache_type,
        preproc=TrainTransform(
                max_labels=50,
                flip_prob=flip_prob,
                hsv_prob=hsv_prob
            ),
    )




def _distillation_loss(student_output, teacher_output, alpha=0.5, temperature=3.0):

    teacher_output_detached = teacher_output.detach()

    # Apply temperature scaling and softmax to both teacher and student outputs
    teacher_probs = F.softmax(teacher_output_detached / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_output['cls_loss'] / temperature, dim=-1)

    # Calculate KL Divergence Loss
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    loss_kl = criterion_kl(student_log_probs, teacher_probs) * (temperature ** 2)

    return loss_kl
    # # Detach teacher outputs to ensure no gradients are propagated through them
    # teacher_output_detached = teacher_output.detach()
    #
    # # Temperature-scaled softmax for the detached teacher output
    # teacher_probs = F.softmax(teacher_output_detached / temperature, dim=-1)
    #
    # # KL Divergence Loss for matching teacher and student class probabilities
    # criterion_kl = nn.KLDivLoss(reduction='batchmean')
    # student_log_probs = F.log_softmax(student_output['cls_loss'] / temperature, dim=-1)
    # loss_kl = criterion_kl(student_log_probs, teacher_probs) * (temperature ** 2)
    #
    # # Additional distillation loss for confidence
    # # Using MSE as an example, adjust according to your task needs
    # criterion_mse = nn.MSELoss()
    # loss_conf = criterion_mse(student_output['conf_loss'], teacher_probs.mean(dim=-1))
    #
    # # Combined loss
    # # Combine the KL divergence loss and the MSE loss for confidence with the student's own total loss
    # combined_loss = alpha * (loss_kl + loss_conf) + (1 - alpha) * student_output['total_loss']
    #
    # return combined_loss

def make_parser():
    parser = argparse.ArgumentParser("YOLOX model distillation")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", "--workspace", type=int, default=32, help="max workspace size in detect"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=1, help="max batch size in detect"
    )
    parser.add_argument(
        "-s", "--samples", type=str, help="path to sample input directory"
    )
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=1,
        help="number of frames for benchmark, rounded up to nearest multiple of batch size",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="path to output directory (not file)"
    )
    parser.add_argument(
        "-t", "--teacher_model_path", type=str, help="path to teacher model path. Must end in .pth"
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        default="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def _get_selected_classes():
    selected_classes = [0,1,2,3,5,7]
    return selected_classes

# Function to filter the dataset
def filter_classes(dataset, classes):
    """ Filter dataset annotations to include only selected classes """
    filtered_dataset = []
    for image, targets in dataset:
        targets = [target for target in targets if target['category_id'] in classes]
        if len(targets) > 0:
            filtered_dataset.append((image, targets))
    return filtered_dataset

def _setup_models(exp, teacher_path, selected_class_indices, device='cuda'):
    # Define the number of classes
    num_classes = len(selected_class_indices)

    # Initialize the teacher model
    teacher_model = exp.get_model()

    full_state_dict = torch.load(teacher_path, map_location="cpu")

    # Filter weights for each class prediction layer that is used in the model
    for key in ['head.cls_preds.0.weight', 'head.cls_preds.1.weight', 'head.cls_preds.2.weight']:
        # Filter weights
        filtered_weights = full_state_dict['model'][key][selected_class_indices, :, :, :]
        full_state_dict['model'][key] = filtered_weights
        # If there are biases associated with these weights
        bias_key = key.replace('weight', 'bias')
        filtered_bias = full_state_dict['model'][bias_key][selected_class_indices]
        full_state_dict['model'][bias_key] = filtered_bias
    teacher_model.load_state_dict(full_state_dict["model"])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Initialize the student model similarly
    student_head = YOLOXHead(num_classes)  # Ensure the student model matches the teacher's setup
    student_model = YOLOX(YOLOPAFPN(), student_head)
    student_model = student_model.to(device)

    return teacher_model, student_model


@logger.catch
@torch.no_grad()
def main():

    args = make_parser().parse_args()

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)
    exp.num_classes = len(_get_selected_classes())

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    teacher_path = 'model/yolox_x.pth'
    class_ids = _get_selected_classes()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    teacher_model, student_model = _setup_models(exp, teacher_path, class_ids, device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    dataset = _get_dataset(exp.test_size, True)
    val_dataset = _get_dataset(exp.test_size, json_file="instances_val2017.json", name="val2017", cache=True)

    best_loss = float('inf')  # Initialize the best loss to a very high value
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(10):  # Number of epochs to train
        # Training phase
        student_model.train()  # Set model to training mode
        teacher_model.eval()  # Ensure the teacher model is in evaluation mode

        for id in range(dataset.num_imgs):
            img, target, img_info, img_id = dataset.__getitem__(id)

            # Ensure img is a torch.Tensor
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img).unsqueeze(0)

            if not isinstance(target, torch.Tensor):
                target = torch.from_numpy(target).unsqueeze(0)

            # Convert img to float and send to the correct device
            img = img.float().to(device)  # Correctly move img to device after converting to float
            target = target.to(device)

            # Compute outputs from both models
            with torch.cuda.amp.autocast(enabled=True):
                teacher_outputs = teacher_model(img)
                student_outputs = student_model(img, target)
                print("Teacher outputs grad_fn:", getattr(teacher_outputs, 'grad_fn', None))
                print("Student outputs grad_fn:", getattr(student_outputs, 'grad_fn', None))
                print(student_outputs)
                loss = student_outputs["total_loss"]
                print("Loss grad_fn before backward:", getattr(loss, 'grad_fn', None))

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                print(f"Epoch {epoch}, Training Loss: {loss.item()}")

        # Validation phase
        student_model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        total_count = 0
        with torch.no_grad():  # No gradients needed for validation
            for id in range(val_dataset.num_images):
                val_labels = torch.tensor(dataset.load_anno(id), device=device, dtype=torch.float32)
                val_images = torch.tensor(dataset.load_resized_img(id), device=device, dtype=torch.float32)
                val_images.to(device)
                val_labels.to(device)

                val_student_outputs = student_model(val_images)
                val_teacher_outputs = teacher_model(val_images)

                val_loss = _distillation_loss(val_student_outputs, val_teacher_outputs, val_labels)
                total_val_loss += val_loss.item() * val_images.size(0)
                total_count += val_images.size(0)

        average_val_loss = total_val_loss / total_count
        print(f"Epoch {epoch}, Validation Loss: {average_val_loss}")

        # Check if the current model is better than the best model seen so far
        if average_val_loss < best_loss:
            best_loss = average_val_loss
            print(f"New best model found at epoch {epoch} with Validation Loss: {average_val_loss}")
            torch.save(student_model.state_dict(), 'best_student_model.pth')  # Save the best model

if __name__ == "__main__":
    main()