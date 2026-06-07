import copy
import math
import os
import random
import sys
import yaml
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from D_AnomalyDetection.Util.Util import anomalydetection_data_loader
from D_AnomalyDetection.Util.Draw_Graph import Draw_Graph


class EfficientNetB4Extractor(nn.Module):
    STAGE_CHANNELS = [32, 56, 160, 448]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        f = efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        ).features
        self.s1 = f[0:3]
        self.s2 = f[3:4]
        self.s3 = f[4:6]
        self.s4 = f[6:8]
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return [f1, f2, f3, f4]


class FeatureAlignModule(nn.Module):
    def __init__(self, in_channels: list, hidden_dim: int = 256, target_hw: tuple = (28, 28)):
        super().__init__()
        self.target_hw = target_hw
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])

    def forward(self, features: list) -> Tensor:
        H, W = self.target_hw
        out = None
        for feat, proj in zip(features, self.projs):
            f = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
            f = proj(f)
            out = f if out is None else out + f
        return out


class PositionEmbeddingSine(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.feature_size  = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        self.scale         = 2 * math.pi

    def forward(self, tensor: Tensor) -> Tensor:
        H, W = self.feature_size
        not_mask = torch.ones((H, W), device=tensor.device)
        y_embed  = not_mask.cumsum(0, dtype=torch.float32)
        x_embed  = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor: Tensor) -> Tensor:
        H, W = self.feature_size
        x_emb = self.col_embed(torch.arange(W, device=tensor.device))
        y_emb = self.row_embed(torch.arange(H, device=tensor.device))
        return torch.cat([
            x_emb.unsqueeze(0).expand(H, -1, -1),
            y_emb.unsqueeze(1).expand(-1, W, -1),
        ], dim=-1).flatten(0, 1)


def build_position_embedding(pos_embed_type: str, feature_size: tuple, hidden_dim: int):
    if pos_embed_type in ("sine", "v2"):
        return PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("learned", "v3"):
        return PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    raise ValueError(f"pos_embed_type: {pos_embed_type}")


def build_neighbor_mask(feature_size: tuple, neighbor_size: tuple) -> Tensor:
    h, w   = feature_size
    hm, wm = neighbor_size
    mask   = torch.ones(h, w, h, w)
    for idx_h1 in range(h):
        for idx_w1 in range(w):
            h2_s = max(idx_h1 - hm // 2, 0)
            h2_e = min(idx_h1 + hm // 2 + 1, h)
            w2_s = max(idx_w1 - wm // 2, 0)
            w2_e = min(idx_w1 + wm // 2 + 1, w)
            mask[idx_h1, idx_w1, h2_s:h2_e, w2_s:w2_e] = 0
    mask = mask.view(h * w, h * w)
    return (mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1   = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2   = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1     = nn.LayerNorm(hidden_dim)
        self.norm2     = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.activation       = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self._with_pos(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src  = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self._with_pos(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src  = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, feature_size, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)
        self.self_attn      = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.norm3   = nn.LayerNorm(hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation       = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, out, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        _, B, _ = memory.shape
        tgt  = self.learned_embed.weight.unsqueeze(1).expand(-1, B, -1)
        tgt2 = self.self_attn(
            query=self._with_pos(tgt, pos), key=self._with_pos(memory, pos), value=memory,
            attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt  = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(
            query=self._with_pos(tgt, pos), key=self._with_pos(out, pos), value=out,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt  = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))

    def forward_pre(self, out, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        _, B, _ = memory.shape
        tgt  = self.learned_embed.weight.unsqueeze(1).expand(-1, B, -1)
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self._with_pos(tgt2, pos), key=self._with_pos(memory, pos), value=memory,
            attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt  = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self._with_pos(tgt2, pos), key=self._with_pos(out, pos), value=out,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt  = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        return tgt + self.dropout3(tgt2)

    def forward(self, out, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(out, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos)
        return self.forward_post(out, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm   = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers              = _get_clones(decoder_layer, num_layers)
        self.norm                = norm
        self.return_intermediate = return_intermediate

    def forward(self, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        out          = memory
        intermediate = []
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(out))
        if self.norm is not None:
            out = self.norm(out)
            if self.return_intermediate:
                intermediate[-1] = out
        return torch.stack(intermediate) if self.return_intermediate else out


class Transformer(nn.Module):
    def __init__(self, hidden_dim, feature_size, neighbor_mask,
                 nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        self.feature_size  = feature_size
        self.neighbor_mask = neighbor_mask
        enc_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        self.encoder = TransformerEncoder(
            enc_layer, num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim) if normalize_before else None,
        )
        dec_layer = TransformerDecoderLayer(
            hidden_dim, feature_size, nhead, dim_feedforward,
            dropout, activation, normalize_before,
        )
        self.decoder = TransformerDecoder(
            dec_layer, num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
            return_intermediate=return_intermediate_dec,
        )
        self._mask_cache: dict = {}

    def _get_mask(self, device) -> Tensor:
        if device not in self._mask_cache:
            ns = self.neighbor_mask["neighbor_size"]
            self._mask_cache[device] = build_neighbor_mask(self.feature_size, ns).to(device)
        return self._mask_cache[device]

    def forward(self, src: Tensor, pos_embed: Tensor):
        _, B, _ = src.shape
        pos = pos_embed.unsqueeze(1).expand(-1, B, -1)
        if self.neighbor_mask:
            mask      = self._get_mask(src.device)
            mask_enc  = mask if self.neighbor_mask["mask"][0] else None
            mask_dec1 = mask if self.neighbor_mask["mask"][1] else None
            mask_dec2 = mask if self.neighbor_mask["mask"][2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None
        memory  = self.encoder(src, mask=mask_enc, pos=pos)
        decoded = self.decoder(memory, tgt_mask=mask_dec1, memory_mask=mask_dec2, pos=pos)
        return decoded, memory


class UniAD(nn.Module):
    def __init__(self, inplanes: list, instrides: list, feature_size: tuple,
                 hidden_dim: int = 256, pos_embed_type: str = "sine",
                 feature_jitter: Optional[dict] = None,
                 neighbor_mask:  Optional[dict] = None,
                 nhead: int = 8, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 normalize_before: bool = False,
                 return_intermediate_dec: bool = False):
        super().__init__()
        assert len(inplanes) == 1 and len(instrides) == 1
        self.feature_size   = feature_size
        self.feature_jitter = feature_jitter
        self.pos_embed   = build_position_embedding(pos_embed_type, feature_size, hidden_dim)
        self.transformer = Transformer(
            hidden_dim=hidden_dim, feature_size=feature_size, neighbor_mask=neighbor_mask,
            nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, normalize_before=normalize_before,
            return_intermediate_dec=return_intermediate_dec,
        )
        self.input_proj  = nn.Linear(inplanes[0], hidden_dim)
        self.output_proj = nn.Linear(hidden_dim,  inplanes[0])
        self.upsample    = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

    def _add_jitter(self, tokens: Tensor) -> Tensor:
        if self.feature_jitter is None:
            return tokens
        scale, prob = self.feature_jitter["scale"], self.feature_jitter["prob"]
        if random.uniform(0, 1) > prob:
            return tokens
        L, B, C = tokens.shape
        norms   = tokens.norm(dim=2, keepdim=True) / C
        return tokens + torch.randn_like(tokens) * norms * scale

    def forward(self, input: dict) -> dict:
        feature_align = input["feature_align"]
        B, C, H, W    = feature_align.shape
        tokens = feature_align.flatten(2).permute(2, 0, 1)
        if self.training and self.feature_jitter:
            tokens = self._add_jitter(tokens)
        tokens    = self.input_proj(tokens)
        pos_embed = self.pos_embed(tokens)
        decoded, _ = self.transformer(tokens, pos_embed)
        feature_rec = self.output_proj(decoded).permute(1, 2, 0).reshape(B, C, H, W)
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )
        pred = self.upsample(pred)
        return {"feature_rec": feature_rec, "feature_align": feature_align, "pred": pred}


class UniADLoss(nn.Module):
    def forward(self, output: dict) -> Tensor:
        return F.mse_loss(output["feature_rec"], output["feature_align"])


class UniADPipeline(nn.Module):
    def __init__(self, align_hidden_dim: int = 256, target_hw: tuple = (28, 28),
                 upsample_stride: int = 8, pos_embed_type: str = "sine",
                 feature_jitter: Optional[dict] = None,
                 neighbor_mask:  Optional[dict] = None,
                 nhead: int = 8, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4, dim_feedforward: int = 2048,
                 dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        self.extractor = EfficientNetB4Extractor(pretrained=pretrained)
        self.align = FeatureAlignModule(
            in_channels=EfficientNetB4Extractor.STAGE_CHANNELS,
            hidden_dim=align_hidden_dim,
            target_hw=target_hw,
        )
        self.uniad = UniAD(
            inplanes=[align_hidden_dim], instrides=[upsample_stride],
            feature_size=target_hw, hidden_dim=align_hidden_dim,
            pos_embed_type=pos_embed_type, feature_jitter=feature_jitter,
            neighbor_mask=neighbor_mask, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.criterion = UniADLoss()

    def forward(self, images: Tensor) -> Tensor:
        features      = self.extractor(images)
        feature_align = self.align(features)
        output        = self.uniad({"feature_align": feature_align})
        return self.criterion(output)

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        self.eval()
        features      = self.extractor(images)
        feature_align = self.align(features)
        output        = self.uniad({"feature_align": feature_align})
        return output["pred"]

    def predict_scores(self, images: Tensor) -> list:
        return self.predict(images).flatten(1).max(dim=1).values.cpu().tolist()


def _save_test_results(model: UniADPipeline, loader, device, threshold, save_dir):
    ok_dir = os.path.join(save_dir, "OK")
    ng_dir = os.path.join(save_dir, "NG")
    os.makedirs(ok_dir, exist_ok=True)
    os.makedirs(ng_dir, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    idx = 0
    correct = 0
    total   = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs  = inputs.to(device)
            scores  = model.predict_scores(inputs)
            imgs    = (inputs.cpu() * std + mean).clamp(0, 1)
            for i, (score, label) in enumerate(zip(scores, labels.tolist())):
                pred   = 1 if score > threshold else 0
                correct += (pred == label)
                total   += 1
                folder  = ng_dir if pred == 1 else ok_dir
                fname   = f"{idx:05d}_pred{'NG' if pred else 'OK'}_gt{'NG' if label else 'OK'}.png"
                to_pil_image(imgs[i]).save(os.path.join(folder, fname))
                idx += 1

    return correct, total


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu":  return F.glu
    raise RuntimeError(f"activation: {activation}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path   = os.path.normpath(os.path.join(current_dir, '..', 'Util', 'config.yaml'))
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_info = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader, test_loader = anomalydetection_data_loader(
        config['load_path'], config['batch_size'], transform_info
    )

    neighbor_mask_cfg  = {"mask": [True, True, True], "neighbor_size": [7, 7]}
    feature_jitter_cfg = {"scale": 20, "prob": 0.5}

    model = UniADPipeline(
        align_hidden_dim   = 256,
        target_hw          = (28, 28),
        upsample_stride    = 8,
        pos_embed_type     = "sine",
        feature_jitter     = feature_jitter_cfg,
        neighbor_mask      = neighbor_mask_cfg,
        nhead              = 8,
        num_encoder_layers = 4,
        num_decoder_layers = 4,
        dim_feedforward    = 2048,
        dropout            = 0.1,
        pretrained         = True,
    ).to(device)

    trainable = list(model.align.parameters()) + list(model.uniad.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epoch']
    )

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])

    epochs         = config['epoch']
    patience       = config['patience']
    patience_count = 0
    best_val_loss  = float('inf')
    threshold      = 0.0

    pbar = tqdm(total=epochs, desc='Total Progress', position=0)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            loss   = model(inputs)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                val_losses.append(model(inputs.to(device)).item())
        val_loss = sum(val_losses) / len(val_losses)

        scores_all, labels_all = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                scores_all.extend(model.predict_scores(inputs.to(device)))
                labels_all.extend(labels.tolist())
        ok_scores = [s for s, l in zip(scores_all, labels_all) if l == 0] or scores_all
        threshold = max(ok_scores)

        epoch_test_dir = os.path.join(config['save_path'], 'val_test', f'epoch_{epoch + 1:04d}')
        correct_val_test, total_val_test = _save_test_results(
            model, test_loader, device, threshold, epoch_test_dir
        )
        val_test_acc = 100 * correct_val_test / total_val_test if total_val_test > 0 else 0.0

        with open(os.path.join(epoch_test_dir, 'result.txt'), 'w') as f:
            f.write(f"Epoch: {epoch + 1}\n"
                    f"Threshold: {threshold:.6f}\n"
                    f"Total: {total_val_test}, Correct: {correct_val_test}\n"
                    f"Accuracy: {val_test_acc:.2f}%\n")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        graph.update_graph(
            train_acc=0, train_loss=train_loss,
            val_acc=val_test_acc, val_loss=val_loss,
            epoch=epoch, patience_count=patience_count,
        )

        pbar.set_postfix({
            'Epoch':      epoch + 1,
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss':   f'{val_loss:.4f}',
            'Threshold':  f'{threshold:.4f}',
            'Test Acc':   f'{val_test_acc:.2f}%',
        })
        pbar.update(1)

    graph.save_plt()
    graph.save_train_info(patience_count)

    best_model_path = os.path.join(config['save_path'], 'Bottom_Loss_Train.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    scores_all, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            scores_all.extend(model.predict_scores(inputs.to(device)))
            labels_all.extend(labels.tolist())
    ok_scores = [s for s, l in zip(scores_all, labels_all) if l == 0] or scores_all
    threshold = max(ok_scores)

    with open(os.path.join(config['save_path'], 'threshold.txt'), 'w') as f:
        f.write(str(threshold))

    final_test_dir = os.path.join(config['save_path'], 'final_test')
    correct_test, total_test = _save_test_results(
        model, test_loader, device, threshold, final_test_dir
    )
    test_acc = 100 * correct_test / total_test if total_test > 0 else 0.0
    print(f'Final Test Accuracy: {test_acc:.2f}%  (threshold={threshold:.6f})')
    graph.save_test_info(total=total_test, correct=correct_test, accuracy=test_acc)


if __name__ == "__main__":
    main()
