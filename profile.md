# Add cpu-profiling option will hang the process
```
# /usr/local/cuda/bin/nvprof --unified-memory-profiling off --cpu-profiling on -f -o prof.nvvp python train.py -c config.json
/usr/local/cuda/bin/nvprof --unified-memory-profiling off -f -o prof.nvvp python train.py -c config.json
python ../nvprof2json/nvprof2json.py ./prof.nvvp > prof.json 

python -m cProfile -o profile.prof train.py -c config.json
```



# Overfit running scripts

```
python train.py -c config.overfit.json
```

```
python test.py -r saved/MyNet/1119_222548/checkpoint-epoch20.pth
```
