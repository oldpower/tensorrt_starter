PDFProcess
===

### runs/train

    names:
        0: chemistry

### runs/train4

    names:
        0: chemistry
        1: reaction


### runs/train5

    names:
        0: chemistry
        1: reaction
        2: others
### app

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python app_demo.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python app_demo.py --workers=4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u app_demo.py --workers=4 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup sh -c 'python -u app_demo.py --workers=4 2>&1 | rotatelogs -n 2 ../logs/run_app_demo.out 10M' &
2943689

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup sh -c 'python -u app_demo.py --workers=4 2>&1 | rotatelogs -n 1 ../logs/run_app_demo.out 10M' > /dev/null 2>&1 &

# app_reaction_extract.py
python -u app_reaction_extract.py > ../logs/run_app_reaction_extract.out 2>&1 &
```

### MinerU

[MinerU](https://github.com/opendatalab/MinerU)

```bash
    export MINERU_MODEL_SOURCE=modelscope
    mineru-api --host 0.0.0.0 --port 8000
    nohup mineru-api --host 0.0.0.0 --port 8000 > ./logs/mineru.out 2>&1 &
```

### rtsp

```bash
sudo apt install apache2-utils

nohup python -u save_rtsp_vedio.py 2>&1 | rotatelogs -n 1 ./logs/run_save_rtsp_vedio.out 100M &

apt install ffmpeg
ffmpeg -decoders | grep -i nvidia
```