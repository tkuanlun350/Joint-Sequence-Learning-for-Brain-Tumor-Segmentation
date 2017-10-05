# Joint-Sequence-Learning-for-Brain-Tumor-Segmentation
Joint Sequence Learning for Brain Tumor Segmentation 

## Dataset
Use dataset/data2record_cmc.py to convert nii file to rfrecord

## Training
python main.py 
    --train_dir=./logs 
    --learning_rate=0.01 
    --model=cmc 

    { ... other configuation on main.py }

## Evaluation
python eval_cmc.py --checkpoint_dir=./logs 

## Update:
    Ensemble results of BRATS-2017 Validation set:

    DiceET: 0.76501	

    DiceWT: 0.88917	

    DiceDT: 0.78167

