## The environment:

- Python 3.6.5 :: Anaconda
- PyTorch 0.4.0
- torchvision 0.2.1
- tensorboardX (for log)
- tensorflow (for visualization)

## To prepare the data:
```shell
bash data-local/bin/prepare_cifar10.sh
```

## To run the code:
```shell
python -m experiments.cifar10_test
```

## Visualization:
Make sure you have installed the tensorflow for tensorboard
```shell
tensorboard --logdir runs
```

