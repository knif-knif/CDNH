cd ..
for nrd in 0 0.01 0.1 0.2 
do
    for nrp in 0.1 0.2 0.3 0.4 0.5
    do
        HF_ENDPOINT=https://hf-mirror.com python3 train_noise.py --dataset=nuswide10k --noiseRate $nrd $nrp --root=/opt/data/private/datasets/ --log-id=La-
    done
done