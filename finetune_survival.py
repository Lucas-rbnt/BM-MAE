import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from bmmae.augmentations import get_train_transforms, get_val_transforms
from bmmae.loops import loop_survival
from bmmae.model import BMMAEViT
from bmmae.tokenizers import MRITokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1ce"],
        help="List of modalities to use for training",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--n_cpus", type=int, default=10, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for the model")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for the model")
    parser.add_argument("--mlp_dim", type=int, default=1536, help="Number of layers for the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads for the model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--entity", type=str, default=None, help="Entity for wandb logging")
    parser.add_argument("--project", type=str, default="bmmae", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs for the model")
    parser.add_argument("--scaling_factor", type=int, default=10, help="Scaling factor for the model")
    parser.add_argument(
        "--from_scratch",
        type=bool,
        default=False,
        help="Whether to train the model from scratch or not",
    )
    parser.add_argument("--froze", type=bool, default=False, help="Whether to freeze the model or not")
    args = parser.parse_args()

    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"Survival_{'-'.join(args.modalities)}_{args.from_scratch}_froze={args.froze}",
            reinit=True,
            config=vars(args),
        )

    set_determinism(seed=args.seed)
    dataframe = pd.read_csv("data/data.csv")
    labtrans = LabTransDiscreteTime(cuts=10, scheme='quantiles')
    labtrans.fit(dataframe["time"].values, dataframe["event"].values)
    trf = labtrans.transform(dataframe["time"].values, dataframe["event"].values)
    dataframe["time_idx"] = trf[0]
    device = "cuda"
    total_c_index = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for j, (train_index, test_index) in enumerate(kfold.split(dataframe, dataframe.time_idx)):
        dataframe_train = dataframe.iloc[train_index]
        dataframe_val = dataframe.iloc[test_index]

        train_files = [
            {modality: os.path.join(path, f"{path.split('/')[-1]}_{modality}.nii.gz") for modality in args.modalities}
            for path in dataframe_train.MRI
        ]
        # add labels, time and event
        for i, path in enumerate(dataframe_train.MRI):
            train_files[i]["time_idx"] = dataframe_train[dataframe_train.MRI == path]["time_idx"].values[0]
            train_files[i]["time"] = dataframe_train[dataframe_train.MRI == path]["time"].values[0]
            train_files[i]["event"] = dataframe_train[dataframe_train.MRI == path]["event"].values[0]

        val_files = [
            {modality: os.path.join(path, f"{path.split('/')[-1]}_{modality}.nii.gz") for modality in args.modalities}
            for path in dataframe_val.MRI
        ]
        # add label
        for i, path in enumerate(dataframe_val.MRI):
            val_files[i]["time_idx"] = dataframe_val[dataframe_val.MRI == path]["time_idx"].values[0]
            val_files[i]["time"] = dataframe_val[dataframe_val.MRI == path]["time"].values[0]
            val_files[i]["event"] = dataframe_val[dataframe_val.MRI == path]["event"].values[0]


        train_dataset = Dataset(data=train_files, transform=get_train_transforms(args.modalities))
        val_dataset = Dataset(data=val_files, transform=get_val_transforms(args.modalities))

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_cpus,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_cpus,
            pin_memory=True,
            persistent_workers=True,
        )

        tokenizers = {
            modality: MRITokenizer(
                patch_size=(args.patch_size, args.patch_size, args.patch_size),
                img_size=(args.img_size, args.img_size, args.img_size),
                hidden_size=args.hidden_size,
            )
            for modality in args.modalities
        }

        model = BMMAEViT(
            modalities=args.modalities,
            tokenizers=tokenizers,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            classification=True,
            n_outputs=10
        )
        if not args.from_scratch:
            print("Using pretrained model")
            state_dict = torch.load("pretrained_models/bmmae_tcga.pth", weights_only=True)
            state_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}
            model.load_state_dict(state_dict, strict=False)
            if args.froze:
                print("Freezing all layers except the last one")
                for name, param in model.named_parameters():
                    if "output_layer" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_func = lambda epoch: min(
            (epoch + 1) / (args.warmup_epochs + 1e-8),
            0.5 * (math.cos(epoch / args.scaling_factor * math.pi) + 1),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

        _, metrics = loop_survival(
            epochs=args.epochs,
            model=model,
            modalities=args.modalities,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dl=train_loader,
            val_dl=val_loader,
            device=device,
            wandb_logging=wandb_logging,
        )
        total_c_index.append(metrics["val/c_index"][-1])

        if args.modalities == ["t1", "t1ce", "t2", "flair"]:
            val_df = dataframe_val.copy()
            predicted_risk = metrics["risk_df"]
            val_df = pd.concat([val_df.reset_index(drop=True), predicted_risk.reset_index(drop=True)], axis=1)
            val_df.to_csv(f"survival_scores/risks_{j}_scratch={args.from_scratch}_froze={args.froze}.csv", index=False)

    total_c_index = np.array(total_c_index)
    logs = {
        "average_c_index": total_c_index.mean(),
        "std_c_index": total_c_index.std(),

    }
    print(logs)
    if wandb_logging:
        wandb.log(logs)
        run.finish()
