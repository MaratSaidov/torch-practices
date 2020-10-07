# Torch Practices
Collecting useful tools &amp; scripts for the development with PyTorch.

To evaluate model size in RAM use `model_size_counter.py`:

```
from model_size_counter import RAMSizeCounter

model = Model()

size_counter = RAMSizeCounter(model, input_size=(16,1,256,256))
print(size_counter.count_megabytes())
```
