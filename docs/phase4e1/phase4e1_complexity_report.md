# Phase 4E-1 Complexity

```json
{
  "full": {
    "parameter_count": 11246081,
    "forward_flops_profiled": 39631676545,
    "profile_batch_size": 8,
    "peak_allocated_bytes_forward_backward": 2745402880
  },
  "no_teacher": {
    "parameter_count": 11246081,
    "forward_flops_profiled": 39648322689,
    "profile_batch_size": 8,
    "peak_allocated_bytes_forward_backward": 2745402880
  },
  "no_forensic": {
    "parameter_count": 11246081,
    "forward_flops_profiled": 25399492608,
    "profile_batch_size": 8,
    "peak_allocated_bytes_forward_backward": 313661440
  },
  "k1": {
    "parameter_count": 10458881,
    "forward_flops_profiled": 39165142145,
    "profile_batch_size": 8,
    "peak_allocated_bytes_forward_backward": 2734706176
  },
  "full_clip": {
    "parameter_count": 11246081,
    "forward_flops_profiled": 39648322689,
    "profile_batch_size": 8,
    "peak_allocated_bytes_forward_backward": 2744838656
  }
}
```

FLOPs 为 profiler 对注明 batch size 的 forward 统计；peak memory 为 integrated forward/backward preflight 的 allocated bytes。
