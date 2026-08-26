# Phase 4E-1 Tier-1 ablations

Full Teacher gate 失败不删除已预注册的 matched Tier-1 结果。可用结果如下；没有 Student summary 的 teacher-dependent arm 是因各自 teacher gate 失败而按协议停止。

```json
{
  "full": {
    "teacher_mask_capability": "FAILED",
    "teacher_selected_mean_fg_iou": 0.26607634745277425,
    "teacher_vs_p1_tf_mean_iou_delta": -0.07685157191185227
  },
  "no_teacher": {
    "student_g0": {
      "n": 1106,
      "mean_foreground_iou": 0.17281810774799355,
      "median_foreground_iou": 0.0976942736368138,
      "mean_foreground_f1": 0.25378384671300164,
      "global_foreground_iou": 0.21735691741932647,
      "global_foreground_f1": 0.35709645102292786,
      "threshold_logit": 0.0
    }
  },
  "no_forensic": {
    "teacher_mask_capability": "FAILED",
    "teacher_selected_mean_fg_iou": 0.22418796975706937,
    "teacher_vs_p1_tf_mean_iou_delta": -0.11873994960755715
  },
  "k1": {
    "teacher_mask_capability": "FAILED",
    "teacher_selected_mean_fg_iou": 0.26379474637064787,
    "teacher_vs_p1_tf_mean_iou_delta": -0.07913317299397865
  },
  "full_clip": {
    "teacher_mask_capability": "FAILED",
    "teacher_selected_mean_fg_iou": 0.2647285275758989,
    "teacher_vs_p1_tf_mean_iou_delta": -0.07819939178872765
  }
}
```
