

## test



### test dit

```shell
parallel=1
url='127.0.0.1'
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

CUDA_VISIBLE_DEVICES=5 python -u test_dit_infer.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree 1 --prompt "一名宇航员在月球上发现一块石碑，上面印有“MindSpore”字样，闪闪发光" --infer_steps 5  --cfg_scale 9.0 --time_shift 13.0 --num_frames 16 --height 128 --width 128
```


### test vae decode

```shell
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

CUDA_VISIBLE_DEVICES=6 python test_vae_decode.py --model_dir $model_dir --ulysses_degree 1
```
