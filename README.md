# PyTorch Implementation of shake-shake

## Usage

```
$ python train.py --depth 26 --base_channels 32 --shake_forward True --shake_backward True --shake_image True --outdir results
```

## Results on CIFAR-10

| Model                        | Test Error (median of 3 runs) | Test Error (in paper)    | Training Time |
|:-----------------------------|:-----------------------------:|:------------------------:|--------------:|
| shake-shake-26 2x32d (S-S-I) | 3.68                          | 3.55 (average of 3 runs) |  33h49m       |
| shake-shake-26 2x64d (S-S-I) | 2.88 (1 run)                  | 2.98 (average of 3 runs) |  78h48m       |
| shake-shake-26 2x96d (S-S-I) | 2.90 (1 run)                  | 2.86 (average of 5 runs) | 101h32m*      |

### Notes

* The model of shake-shake-26 2x64d (S-S-I) is trained with batch size 64, and initial learning rate 0.1.

![](figures/shake-shake-26_2x32d.png)
![](figures/shake-shake-26_2x64d.png)
![](figures/shake-shake-26_2x96d.png)

## References

* Gastaldi, Xavier. "Shake-Shake regularization." In International Conference on Learning Representations, 2017. [arXiv:1705.07485]( https://arxiv.org/abs/1705.07485 ), [Torch implementation]( https://github.com/xgastaldi/shake-shake )


