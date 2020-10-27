# CTPN + CRNN  Pytorch Chinese OCR

Chinese characters recognition repository based on CTPN， CRNN



## Performance

recognize Chinese Characters from image, here shows some results

<p align='center'>
<img src='output/result/result_0.png' title='result0' style='max-width:600px'></img>
</p>
<p align='center'>
<img src='output/result/result_1.png' title='result1' style='max-width:600px'></img>
</p>

## Environments

1. WIN 10 or Ubuntu 16.04
2. **PyTorch 1.2.0 (may fix ctc loss)** with cuda 10.0，**recommend 1.6.0**
3. yaml
4. easydict
5. tensorboardX
6. opencv
7. pillow
8. numpy

##  How to use?

```
[run] python ctpn_crnn.py --image_path images/test_1.png (or change to your image path)
```

after run above, you could fine result from folder

```
output/result/result_0.png (or result_1.png depending on how many image you had run)
```



## File instructions

- crnn  (folder)
  - include all  CRNN scripts
- ctpn_pytorch (folder)
  - include all CTPN scripts
- font (folder)
  - include some fonts support Chinese
- images (folder)
  - include some test images
- output (folder)
  - chechkpoints (folder)
    - include CRNN checkpoint
  - result (folder)
    - include test output
- Download CRNN pretrained weights [here](https://pan.baidu.com/s/1pKfmZO9LFpLmUokXpcOPPQ) (password: kze0)
  - copy to **output/checkpoints/**
- Download CTPN pretrained weights [here](https://pan.baidu.com/s/16gBfh6Uq0eQYsKPAjVpv-A) (password: 5bpi)
  - copy to **ctpn_pytorch/weights/**

## How to train & Data set

###  Train CRNN model

1. Data set for **CRNN** training

   - 3.6 Million Synthetic Chinese String，download from [here](https://pan.baidu.com/s/1ErLFLUf8IFTDnzxAs8parA ) , (password: auwl)

2. Edit **crnn/lib/config/360cc_config.yaml DATA:ROOT**  to your image path

   - ```
     DATASET:360CC
     ROOT: "your image path"
     ```

3. Download [labels](https://pan.baidu.com/s/11rUYMON7FqI8u-dAIjARyw) (password: fwup)

4. Put *char_std_5990.txt* in **lib/dataset/txt/**

5. And put *train.txt* and *test.txt* in **lib/dataset/txt/**

   eg. test.txt

   ```
   20456343_4045240981.jpg 89 201 241 178 19 94 19 22 26 656
   20457281_3395886438.jpg 120 1061 2 376 78 249 272 272 120 1061
   ......
   ```

   #### Or your own data

   1. Edit **crnn/lib/config/OWN_config.yaml** DATA:ROOT to you image path

   ```angular2html
       DATASET:
         ROOT: 'to/your/images/path'
   ```

   2. And put your *train_own.txt* and *test_own.txt* in **crnn/lib/dataset/txt/**

      eg. test_own.txt

   ```
       20456343_4045240981.jpg 你好啊！祖国！
       20457281_3395886438.jpg 晚安啊！世界！
       ...
   ```

   **note**: fixed-length training is supported. yet you can modify dataloader to support random length training.   

6. **Train**

   ```
   [run] cd crnn
   [run] python train.py --cfg lib/config/360CC_config.yaml
   or
   [run] python train.py --cfg lib/config/OWN_config.yaml
   ```

   ```
   loss curve
      [run] cd output/360CC/crnn/xxxx-xx-xx-xx-xx/
      [run] tensorboard --logdir log
   ```

7. **Test**

   ```
   [run] cd crnn
   [run] python demo.py --image_path images/test.png --checkpoint output/checkpoints/mixed_second_finetune_acc_97P7.pth
   ```

   

### Train CTPN model

1. Data set for CTPN training，download [here](https://pan.baidu.com/s/1TF_CZI9Vt5L-Wq2HYZlF2w) (password: ffoq)

   this dataset is VOC2007 format

   ```
   --VOC2007
    -- Annotations
    -- ImageSets
    -- JPEGImages
   ```

2. Edit **ctpn_pytorch/ctpn/config.py**

   ```
   img_dir = 'VOC2007/JPEGImages/'
   label_dir = 'VOC2007/Annotations/'
   ```

3. **Train**

   ```
   cd ctpn_pytorch
   python train.py
   ```

4. **Test**

   ```
   cd ctpn_pytorch
   python predict.py
   ```



## References

https://github.com/xiaofengShi/CHINESE-OCR

https://github.com/CrazySummerday/ctpn.pytorch