cd ..
for nrd in 0 0.01 0.1 0.2 
do
    for nrp in 0.1 0.2 0.3 0.4 0.5
    do
        python3 train_noise.py --dataset=nuswide10k --noiseRate $nrd $nrp --root=/opt/data/private/datasets/ --arch=ViT --warm-up=5 --log-id=Clean-
    done
done