import argparse
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from run_logging import configure_run_logging


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="Imagenette2")
parser.add_argument("--attack_method", type=str, default="BadNet")
parser.add_argument("--dataset_config", type=str, default="imagenet_256.yml", help="Path to the config file")
parser.add_argument("--attack_schedule", type=str, default="BadNet", help="Path to the config file")
parser.add_argument("--datasets_root_dir", type=str, default="datasets")
parser.add_argument("--classes", type=int, default=20)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--attack_target", type=int, default=1)
parser.add_argument("--poisoned_rate", type=float, default=0.05)
parser.add_argument("--lc_adv_dataset_dir", type=str, default="./adv_dataset")
parser.add_argument("--lc_adv_model_ckpt", type=str, default="")
parser.add_argument("--lc_adv_batch_size", type=int, default=16)
parser.add_argument("--lc_adv_num_workers", type=int, default=2)
parser.add_argument("--lc_pgd_steps", type=int, default=100)
parser.add_argument("--issba_secret_size", type=int, default=20)
parser.add_argument("--issba_exported_poisoned_root", type=str, default="")
parser.add_argument("--iad_cross_rate", type=float, default=0.1)
parser.add_argument("--iad_lambda_div", type=float, default=1.0)
parser.add_argument("--iad_lambda_norm", type=float, default=100.0)
parser.add_argument("--iad_mask_density", type=float, default=0.032)
parser.add_argument("--adaptive_patch_covered_rate", type=float, default=0.05)

parser.add_argument("--testwclean", type=str2bool, default=False)
parser.add_argument("--testwpoisoned", type=str2bool, default=False)
parser.add_argument("--testwpurified", type=str2bool, default=True)
parser.add_argument("-ptra", "--purify_traindataset", action="store_true")
parser.add_argument("-pctes", "--purify_clean_test", action="store_true")
parser.add_argument("-pptes", "--purify_pois_test", action="store_true")
parser.add_argument("-uptra", "--use_purified_train", action="store_true")
parser.add_argument("-upctes", "--use_purified_clean_test", action="store_true")
parser.add_argument("-upptes", "--use_purified_pois_test", action="store_true")
parser.add_argument("--mode", type=int, default=0)
parser.add_argument("--concat", type=str2bool, default=True)

parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
parser.add_argument("--deg", type=str, default="gaussian_noise", help="Degradation name")
parser.add_argument("--path_y", type=str, default="attack", help="Path of the test dataset.")
parser.add_argument("--exp", type=str, default="exp", help="Path for saved pre-trained diffusion model.")
parser.add_argument("--at_threshold", type=int, default=10, help="at_threshold")
parser.add_argument("--subset_start", type=int, default=0, help="Starting index used for ZIP purified image filenames")
parser.add_argument("--simplified", default="True", help="Use simplified DDNM, without SVD")
parser.add_argument("--image_folder", type=str, default="images", help="The folder name of samples")
parser.add_argument("--image_test_folder", type=str, default="images", help="The folder name of samples")
parser.add_argument("--image_test_folder_pois", type=str, default="images", help="The folder name of samples")
parser.add_argument("--splited_image_folder", type=str, default="images", help="The folder name of samples")
parser.add_argument("--splited_test_image_folder", type=str, default="images", help="The folder name of samples")
parser.add_argument("--splited_test_image_folder_pois", type=str, default="images", help="The folder name of samples")
parser.add_argument("--apy_folder", type=str, default="./apy", help="The folder name of samples with linear transformation")
parser.add_argument("--deg_scale", type=float, default=0.1, help="Degradation strength")
parser.add_argument("--ni", default=True, help="No interaction")
parser.add_argument("--deterministic", type=str2bool, default=True)
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--sampling", type=int, default=20)

parser.add_argument("--pur_folder", type=str, default="./pur", help="The folder name of purified images samples")
parser.add_argument("--splited_pur_folder", type=str, default="./pur_splited", help="The folder name of splited image samples")
parser.add_argument("--dataset_seed", type=int, default=42)
parser.add_argument("--gpu", type=str, default="cuda:0")
parser.add_argument("--gpulist", nargs="+", type=int, default=[0])
parser.add_argument("--useAVGUP", type=str2bool, default=False)
parser.add_argument(
    "--purified_label_style",
    type=str,
    default="class_name",
    choices=["class_name", "numeric"],
    help="Folder label style for newly exported purified images.",
)

parser.add_argument(
    "--purifier_backend",
    type=str,
    default="convir_zip",
    choices=["zip", "var_deg_zip", "diffusionzip", "convir_zip", "sampdetox", "none"],
)
parser.add_argument("--zip_config", type=str, default="imagenet_256.yml")
parser.add_argument("--zip_timesteps", type=int, default=1000)
parser.add_argument("--zip_sampling", type=int, default=20)
parser.add_argument(
    "--zip_operator",
    type=str,
    default="sr_averagepooling",
    choices=["sr_averagepooling", "identity", "gaussian_blur"],
    help="Second ZIP guidance operator used by purifier_backend=var_deg_zip.",
)
parser.add_argument("--zip_blur_kernel", type=int, default=5, help="Odd Gaussian blur kernel for zip_operator=gaussian_blur.")
parser.add_argument("--zip_blur_sigma", type=float, default=1.0, help="Gaussian blur sigma for zip_operator=gaussian_blur.")
parser.add_argument(
    "--diffusionzip_start_t",
    type=int,
    default=300,
    help="Actual diffusion timestep used to noise the input before diffusionzip reverse sampling.",
)
parser.add_argument("--degradation_type", type=str, default="gaussian_noise")
parser.add_argument("--degradation_strength", type=float, default=0.1)
parser.add_argument("--sampdetox_ckpt", type=str, default="")
parser.add_argument("--sampdetox_config", type=str, default="")
parser.add_argument("--sampdetox_source_dir", type=str, default="../SampDetox")
parser.add_argument("--sampdetox_image_size", type=int, default=32)
parser.add_argument("--sampdetox_t1", type=int, default=120)
parser.add_argument("--sampdetox_t2", type=int, default=120)
parser.add_argument("--sampdetox_num_blocks", type=int, default=8)
parser.add_argument("--sampdetox_save_diagnostics", type=str2bool, default=False)
parser.add_argument("--use_conv_ir", action="store_true", default=False)
parser.add_argument("--disable_conv_ir", action="store_true", default=False)
parser.add_argument("--restorer_ckpt", type=str, default="./pretrain/ots-base.pkl")
parser.add_argument("--restorer_version", type=str, default="base")
parser.add_argument("--log_dir", type=str, default="logs", help="Directory for run-level console logs")
parser.add_argument("--log_file", type=str, default="", help="Optional explicit run log file path")
parser.add_argument("--disable_run_log", action="store_true", default=False)
parser.set_defaults(purify_train=False, purify_poisoned_test=False)

base_args = parser.parse_args()
if base_args.purifier_backend in {"zip", "var_deg_zip", "diffusionzip"}:
    base_args.dataset_config = base_args.zip_config

with open(os.path.join("configs", base_args.dataset_config), "r") as f:
    config = yaml.safe_load(f)
base_config = dict2namespace(config)


def _safe_path_part(value: object) -> str:
    return str(value).strip().replace("\\", "_").replace("/", "_").replace(":", "_").replace(" ", "_")


def _configure_console_log(args) -> None:
    if args.disable_run_log:
        return
    if args.log_file:
        log_path = args.log_file
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = "_".join(
            [
                timestamp,
                _safe_path_part(args.dataset),
                _safe_path_part(args.attack_method),
                _safe_path_part(args.deg),
                _safe_path_part(args.deg_scale),
                f"seed{_safe_path_part(args.seed)}",
            ]
        )
        log_path = os.path.join(args.log_dir, f"{filename}.log")
    args.run_log_path = configure_run_logging(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info("Run log: %s", args.run_log_path)


_configure_console_log(base_args)

print(f"Using the {base_args.dataset}")
print(f"The at_threshold: {base_args.at_threshold}")


def _sync_flag_aliases(args):
    args.purify_train = args.purify_traindataset
    args.purify_poisoned_test = args.purify_pois_test

    default_deg = parser.get_default("deg")
    default_degradation_type = parser.get_default("degradation_type")
    default_deg_scale = parser.get_default("deg_scale")
    default_degradation_strength = parser.get_default("degradation_strength")
    if args.deg == default_deg and args.degradation_type != default_degradation_type:
        args.deg = args.degradation_type
    elif args.degradation_type == default_degradation_type and args.deg != default_deg:
        args.degradation_type = args.deg
    if args.deg_scale == default_deg_scale and args.degradation_strength != default_degradation_strength:
        args.deg_scale = args.degradation_strength
    elif args.degradation_strength == default_degradation_strength and args.deg_scale != default_deg_scale:
        args.degradation_strength = args.deg_scale

    if args.purifier_backend in {"zip", "var_deg_zip", "diffusionzip"}:
        args.dataset_config = args.zip_config
        args.timesteps = args.zip_timesteps
        args.sampling = args.zip_sampling
        args.use_conv_ir = False
    elif args.purifier_backend == "convir_zip":
        args.use_conv_ir = not args.disable_conv_ir
    elif args.disable_conv_ir:
        args.use_conv_ir = False


def _set_mode(args):
    if args.use_purified_train:
        if args.use_purified_clean_test and args.use_purified_pois_test:
            args.mode = 1
        elif not args.use_purified_clean_test and not args.use_purified_pois_test:
            args.mode = 2
    else:
        if args.use_purified_clean_test and args.use_purified_pois_test:
            args.mode = 3
        elif not args.use_purified_clean_test and not args.use_purified_pois_test:
            args.mode = 4


def _infer_num_classes(dataset_name: str) -> Optional[int]:
    normalized = dataset_name.strip().lower()
    mapping = {
        "cifar10": 10,
        "cifar-10": 10,
        "gtsrb": 43,
    }
    return mapping.get(normalized)


def _prepare_output_dirs(args):
    base_dir = os.path.join(
        args.pur_folder,
        f"Mode{args.mode}",
        f"{args.dataset}",
        f"{args.attack_method}",
        f"{args.deg}",
        f"{args.deg_scale}",
        f"{args.at_threshold}",
    )
    split_base_dir = os.path.join(
        args.splited_pur_folder,
        f"Mode{args.mode}",
        f"{args.dataset}",
        f"{args.attack_method}",
        f"{args.deg}",
        f"{args.deg_scale}",
        f"{args.at_threshold}",
    )

    args.image_folder = os.path.join(base_dir, "train")
    args.test_image_folder = os.path.join(base_dir, "val")
    args.test_image_folder_pois = os.path.join(base_dir, "val_pois")
    args.splited_image_folder = os.path.join(split_base_dir, "train")
    args.splited_test_image_folder = os.path.join(split_base_dir, "val")
    args.splited_test_image_folder_pois = os.path.join(split_base_dir, "val_pois")

    args.train_image_folder_apy = os.path.join(base_dir, "train_apy")
    args.train_image_folder_orig = os.path.join(base_dir, "train_orig")
    args.test_image_folder_apy = os.path.join(base_dir, "val_apy")
    args.test_image_folder_orig = os.path.join(base_dir, "val_orig")
    args.test_pois_image_folder_apy = os.path.join(base_dir, "val_pois_apy")
    args.test_pois_image_folder_orig = os.path.join(base_dir, "val_pois_orig")

    for folder in [
        args.image_folder,
        args.test_image_folder,
        args.test_image_folder_pois,
        args.splited_image_folder,
        args.splited_test_image_folder,
        args.splited_test_image_folder_pois,
        args.train_image_folder_apy,
        args.train_image_folder_orig,
        args.test_image_folder_apy,
        args.test_image_folder_orig,
        args.test_pois_image_folder_apy,
        args.test_pois_image_folder_orig,
    ]:
        os.makedirs(folder, exist_ok=True)


def _print_run_settings(args, device: torch.device) -> None:
    config_rows = [
        ("dataset", args.dataset),
        ("dataset_config", args.dataset_config),
        ("attack_method", args.attack_method),
        ("attack_schedule", args.attack_schedule),
        ("attack_target", args.attack_target),
        ("poisoned_rate", args.poisoned_rate),
        ("lc_adv_dataset_dir", args.lc_adv_dataset_dir),
        ("lc_adv_model_ckpt", args.lc_adv_model_ckpt),
        ("lc_adv_batch_size", args.lc_adv_batch_size),
        ("lc_adv_num_workers", args.lc_adv_num_workers),
        ("lc_pgd_steps", args.lc_pgd_steps),
        ("issba_secret_size", args.issba_secret_size),
        ("issba_exported_poisoned_root", args.issba_exported_poisoned_root),
        ("iad_cross_rate", args.iad_cross_rate),
        ("iad_lambda_div", args.iad_lambda_div),
        ("iad_lambda_norm", args.iad_lambda_norm),
        ("iad_mask_density", args.iad_mask_density),
        ("adaptive_patch_covered_rate", args.adaptive_patch_covered_rate),
        ("img_size", args.img_size),
        ("seed", args.seed),
        ("dataset_seed", args.dataset_seed),
        ("deterministic", args.deterministic),
        ("deg", args.deg),
        ("deg_scale", args.deg_scale),
        ("purify_train", args.purify_train),
        ("purify_clean_test", args.purify_clean_test),
        ("purify_poisoned_test", args.purify_poisoned_test),
        ("use_purified_train", args.use_purified_train),
        ("use_purified_clean_test", args.use_purified_clean_test),
        ("use_purified_pois_test", args.use_purified_pois_test),
        ("testwclean", args.testwclean),
        ("testwpoisoned", args.testwpoisoned),
        ("testwpurified", args.testwpurified),
        ("concat", args.concat),
        ("at_threshold", args.at_threshold),
        ("subset_start", args.subset_start),
        ("timesteps", args.timesteps),
        ("sampling", args.sampling),
        ("gpu", args.gpu),
        ("gpulist", args.gpulist),
        ("useAVGUP", args.useAVGUP),
        ("purified_label_style", args.purified_label_style),
        ("purifier_backend", args.purifier_backend),
        ("zip_config", args.zip_config),
        ("zip_timesteps", args.zip_timesteps),
        ("zip_sampling", args.zip_sampling),
        ("zip_operator", args.zip_operator),
        ("zip_blur_kernel", args.zip_blur_kernel),
        ("zip_blur_sigma", args.zip_blur_sigma),
        ("diffusionzip_start_t", args.diffusionzip_start_t),
        ("degradation_type", args.degradation_type),
        ("degradation_strength", args.degradation_strength),
        ("sampdetox_ckpt", args.sampdetox_ckpt),
        ("sampdetox_config", args.sampdetox_config),
        ("sampdetox_source_dir", args.sampdetox_source_dir),
        ("sampdetox_image_size", args.sampdetox_image_size),
        ("sampdetox_t1", args.sampdetox_t1),
        ("sampdetox_t2", args.sampdetox_t2),
        ("sampdetox_num_blocks", args.sampdetox_num_blocks),
        ("sampdetox_save_diagnostics", args.sampdetox_save_diagnostics),
        ("use_conv_ir", args.use_conv_ir),
        ("disable_conv_ir", args.disable_conv_ir),
        ("restorer_ckpt", args.restorer_ckpt),
        ("restorer_version", args.restorer_version),
        ("log_dir", args.log_dir),
        ("log_file", args.log_file),
        ("disable_run_log", args.disable_run_log),
    ]
    runtime_rows = [
        ("mode", args.mode),
        ("classes", args.classes),
        ("device", device),
        ("run_log_path", getattr(args, "run_log_path", "<disabled>")),
    ]
    non_default_rows = [
        (key, value)
        for key, value in config_rows
        if parser.get_default(key) != value
    ]

    print("=" * 84)
    print("Non-default run settings")
    print("=" * 84)
    if non_default_rows:
        for key, value in non_default_rows:
            print(f"{key}: {value}")
    else:
        print("<none>")
    print("-" * 84)
    print("Runtime metadata")
    print("-" * 84)
    for key, value in runtime_rows:
        print(f"{key}: {value}")
    print("=" * 84)


def parse_args_and_config(args, new_config):
    _sync_flag_aliases(args)
    _set_mode(args)
    default_classes = parser.get_default("classes")
    if args.classes == default_classes:
        inferred_classes = _infer_num_classes(args.dataset)
        if inferred_classes is not None:
            args.classes = inferred_classes
    _prepare_output_dirs(args)

    device = torch.device(f"{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: %s", device)
    new_config.device = device
    _print_run_settings(args, device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = not bool(args.deterministic)
    if bool(args.deterministic):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

    return args, new_config
