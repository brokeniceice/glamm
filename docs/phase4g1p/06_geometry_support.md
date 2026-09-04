# Geometry and support contract

`F24,z_F24` 均使用 Phase4G-0.5 已验证的 CLIP crop cell-center → original-normalized mapping，在64-grid形成 `F64,z_F64,M64`，在256-grid形成 forensic masses/support。两个 tensor共享同一 geometry object，不允许独立 resize假设。

Outside support：

- ordinary context feature/logit只以0作 storage placeholder；
- forensic evidential mass严格 `[0,0,1]`（vacuous），不是background certainty；
- `Fctx64=0`；
- comparison与 `U_F64=0`；
- `U_F256=0`；
- final output逐像素 direct dispatch `z_L`。

Synthetic non-square geometry `[resized 256×320, crop yxyx 16,48,240,272]` 验证 support mapping、outside utility zero与vacuous mass全部通过。Fully unsupported crop触发 whole-image exact P1 fallback。`GEOMETRY_SUPPORT=PASS`。
