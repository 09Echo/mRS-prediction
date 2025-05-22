# import json
# import torch
# from open_clip import create_model_and_transforms, get_tokenizer
# from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
#
# # Load the model and config files
# model_name = "biomedclip_local"
#
# with open("checkpoint/open_clip_config.json", "r") as f:
#     config = json.load(f)
#     model_cfg = config["model_cfg"]
#     preprocess_cfg = config["preprocess_cfg"]
#
#
# if (not model_name.startswith(HF_HUB_PREFIX)
#     and model_name not in _MODEL_CONFIGS
#     and config is not None):
#     _MODEL_CONFIGS[model_name] = model_cfg
#
# tokenizer = get_tokenizer(model_name)
#
# model, _, preprocess = create_model_and_transforms(
#     model_name=model_name,
#     pretrained="checkpoint/open_clip_pytorch_model.bin",
#     **{f"image_{k}": v for k, v in preprocess_cfg.items()},
# )
#
# model.eval()
#
# images = torch.randn(1,3,224,224)
# texts = tokenizer('i am ok!')
# image_features, text_features, logit_scale = model(images, texts)
# print(image_features.shape)
# print(text_features.shape)


# import torch.nn as nn
# from open_clip import create_model_and_transforms, get_tokenizer
# import json
# import torch
# from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
#
# class BiomedClipModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         with open('/home/hubin/Codes/ISLES2024/image_text_models/biomedclip/checkpoint/open_clip_config.json', "r") as f:
#             config = json.load(f)
#             model_cfg = config["model_cfg"]
#             preprocess_cfg = config["preprocess_cfg"]
#
#         model_name="biomedclip_local"
#         if (not model_name.startswith(HF_HUB_PREFIX)
#             and model_name not in _MODEL_CONFIGS
#             and config is not None):
#             _MODEL_CONFIGS[model_name] = model_cfg
#
#         self.tokenizer = get_tokenizer("biomedclip_local")
#
#         # Load model
#         self.model, _, self.preprocess = create_model_and_transforms(
#             model_name="biomedclip_local",
#             pretrained='/home/hubin/Codes/ISLES2024/image_text_models/biomedclip/checkpoint/open_clip_pytorch_model.bin',
#             **{f"image_{k}": v for k, v in preprocess_cfg.items()},
#         )
#
#         self.fc = nn.Linear(512*2, 2)
#
#         for params in self.model.parameters():
#             params.requires_grad = False
#
#
#     def forward(self, images, texts):
#         texts = self.tokenizer(texts)
#         image_features, text_features, logit_scale = self.model(images, texts)
#         feature = torch.concatenate((image_features,text_features),dim=1)
#         out = self.fc(feature)
#
#         return out
#
# if __name__ == '__main__':
#     images = torch.randn(1,3,224,224)
#     texts = "i am ok!"
#     # model_name = "biomedclip_local"
#     # config_path = "checkpoint/open_clip_config.json"
#     # checkpoint_path = "checkpoint/open_clip_pytorch_model.bin"
#
#     model = BiomedClipModel()
#     out = model(images, texts)
#     print(out.shape)
#     total = sum(param.numel() for param in model.parameters() if param.requires_grad)
#     print('  + Number of Backbone Params: %.4f(e6)' % (total / 1e6))
#     for param_name, param in model.named_parameters():
#         print(param_name, param.requires_grad)

import torch.nn as nn
from open_clip import create_model_and_transforms, get_tokenizer
import json
import torch
from peft import LoraConfig, get_peft_model
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


class BiomedClipModel(nn.Module):
    def __init__(self):
        super().__init__()

        with open('/home/hubin/Codes/ISLES2024/image_text_models/biomedclip/checkpoint/open_clip_config.json',
                  "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        model_name = "biomedclip_local"
        if (not model_name.startswith(HF_HUB_PREFIX)
                and model_name not in _MODEL_CONFIGS
                and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        self.tokenizer = get_tokenizer("biomedclip_local")

        # Load model
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name="biomedclip_local",
            pretrained='/home/hubin/Codes/ISLES2024/image_text_models/biomedclip/checkpoint/open_clip_pytorch_model.bin',
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        # 冻结除 LoRA 以外的所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 定义 LoRA 配置
        lora_config = LoraConfig(
            r=8,  # 低秩维度，影响参数量
            target_modules=["proj"],  # 只对 `Linear` 层的 `proj` 部分应用 LoRA
            lora_dropout=0.1,  # Dropout 防止过拟合
            bias="none",
        )

        # 仅对 `visual` 模块应用 LoRA
        self.model.visual = get_peft_model(self.model.visual, lora_config)

        self.fc = nn.Linear(512 * 2, 2)

    def forward(self, images, texts):
        texts = self.tokenizer(texts).cuda()
        image_features, text_features, logit_scale = self.model(images, texts)

        feature = torch.cat((image_features, text_features), dim=1)
        out = self.fc(feature)

        return out


if __name__ == '__main__':
    images = torch.randn(1, 3, 224, 224)
    texts = "i am ok!"

    model = BiomedClipModel()

    out = model(images, texts)
    print(out.shape)

    # 统计 LoRA 训练参数
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  + Number of LoRA Trainable Params: {total_trainable_params / 1e6:.4f}M")

    for name, param in model.named_parameters():
        print(f"{name}: Requires Grad = {param.requires_grad}")
