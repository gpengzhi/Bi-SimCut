# Bi-SimCut: A Simple Strategy for Boosting Neural Machine Translation

This repository contains the PyTorch implementation for our NAACL 2022 main conference paper "[Bi-SimCut: A Simple Strategy for Boosting Neural Machine Translation](https://aclanthology.org/2022.naacl-main.289/)". Please check [here]() for the PaddlePaddle implementation for our work.

## Requirements and Installation

This work has been tested in the following environment.

* Python version == 3.6.5
* PyTorch version == 1.10.1
* Fairseq version == 0.12.2

To install fairseq and develop locally:
```
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

## Bi-SimCut

The following instructions can be used to train a Transformer model on the IWSLT'14 German to English dataset.

Download and preprocess the data:
```
# Download and prepare the unidirectional data
bash prepare-iwslt14.sh

# Preprocess/binarize the unidirectional data
TEXT=./iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --joined-dictionary --workers 20

# Prepare the bidirectional data
cd iwslt14.tokenized.de-en
cat train.en train.de > train.src
cat train.de train.en > train.tgt
cat valid.en valid.de > valid.src
cat valid.de valid.en > valid.tgt
cat test.en test.de > test.src
cat test.de test.en > test.tgt
cd ..

# Preprocess/binarize the bidirectional data
TEXT=./iwslt14.tokenized.de-en
fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.bidirection.de-en \
    --srcdict data-bin/iwslt14.tokenized.de-en/dict.en.txt --tgtdict data-bin/iwslt14.tokenized.de-en/dict.de.txt --workers 20
```

Pretrain a Transformer translation model over the bidirectional data:
```
EXP=iwslt14_de_en_bid_simcut_alpha3_p005
DATA=data-bin/iwslt14.tokenized.bidirection.de-en

mkdir -p checkpoint/$EXP
mkdir -p log/$EXP

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_simcut --alpha 3.0 --p 0.05 --label-smoothing 0.1 \
    --max-tokens 4096 --fp16 --no-epoch-checkpoints --save-dir checkpoint/$EXP \
    1>log/$EXP/log.out 2>log/$EXP/log.err
```

Finetune a Transformer translation model over the unidirectional data:
```
EXP=iwslt14_de_en_simcut_alpha3_p005
DATA=data-bin/iwslt14.tokenized.de-en
CKPT=checkpoint/iwslt14_de_en_bid_simcut_alpha3_p005/checkpoint_best.pt

mkdir -p checkpoint/$EXP
mkdir -p log/$EXP

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_simcut --alpha 3.0 --p 0.05 --label-smoothing 0.1 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --restore-file $CKPT \
    --max-tokens 4096 --fp16 --no-epoch-checkpoints --save-dir checkpoint/$EXP \
    1>log/$EXP/log.out 2>log/$EXP/log.err
```

Evaluate our trained model:
```
DATA=data-bin/iwslt14.tokenized.de-en
REF=iwslt14.tokenized.de-en/test.en
EXP=iwslt14_de_en_simcut_alpha3_p005
CKPT=checkpoint/$EXP/checkpoint_best.pt

mkdir -p evaluation

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --path $CKPT \
    --gen-subset test --beam 5 --lenpen 1 --max-tokens 8192 --remove-bpe
    > evaluation/$EXP

FILE=evaluation/$EXP

cat $FILE | grep -P "^D" | sort -V | cut -f 3- > $FILE.detok
sed -r 's/(@@ )|(@@ ?$)//g' $REF > $REF.detok

MOSES=mosesdecoder
BLEU=$MOSES/scripts/generic/multi-bleu.perl

perl $BLEU $REF.detok < $FILE.detok
```


## Citation

If you find the resources in this repository helpful, please cite as:
```
@inproceedings{gao-etal-2022-bi,
    title = "{B}i-{S}im{C}ut: A Simple Strategy for Boosting Neural Machine Translation",
    author = "Gao, Pengzhi  and He, Zhongjun  and Wu, Hua  and Wang, Haifeng",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.289",
    pages = "3938--3948",
}
```
