# Geometry 与 support hardening

## 坐标契约

所有 fusion 位置统一定义为 original-image normalized cell center：目标格 `(H,W)` 上的位置是 `((x+0.5)/W,(y+0.5)/H)`。

- `S64` 与 SAM raw low-res tensor先移除 right/bottom padding，再表达在 original-normalized grid。
- `F24/z_F24` 通过 CLIP resize-shortest + center-crop 的真实 `resized_hw` 与 `crop_box_yxyx` 反投影。
- `grid_sample` 使用 bilinear、`align_corners=False`。
- support 由目标 cell center是否落在 crop box内定义，不由 feature 是否为 0 推断。

## Outside-support 语义

CLIP center-crop 外严格设为 vacuous mass：

```text
m({background}) = 0
m({foreground}) = 0
m(Omega) = 1
```

zero feature/logit 仅可作为存储 placeholder，不能进入 reliability 语义，不能成为 reliable negative evidence。

## 测试结果

square image support fraction为 1；`400×800` 与 `800×400` 的合成 center-crop support fraction均为 0.5，center supported、corners unsupported且均为 exact vacuous。mass normalization最大误差为 `1.1920928955078125e-7`。SAM `512×1024` resize + bottom padding案例正确移除 padding。

cross-image 固定 target image geometry与support，只替换 forensic content；spatial shuffle只打乱 supported forensic value，不打乱 support mask。language tensor在两种 corruption下不变。

```yaml
GEOMETRY_VALIDATED: YES
VACUOUS_SUPPORT_VALIDATED: YES
```

机器可读结果见 `outputs/phase4g05/geometry_support.json`。

