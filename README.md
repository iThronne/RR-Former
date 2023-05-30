# RR-Former: Rainfall-runoff modeling based on Transformer

This is our first version of RR-Former in the following paper: [RR-Former: Rainfall-runoff modeling based on Transformer](https://www.sciencedirect.com/science/article/abs/pii/S0022169422003560). Thank you for your attention to our work. I'll release more elegant code in the future version. 

1. Using *pretrain.py* to pretrain a RR-Former. Furthermore, using *pretrain_test_global.py* to test the pretrained model in a global way, and using *pretrain_test_single.py* to test the pretrained model in a single-basin way.
2. Using *fine_tune.py* to fine tune a RR-Former model (based on pretrained RR-Former).  Furthermore, using *fine_tune_test.py* to test the fine-tuned model.

## Notice

- Download data from the following paper: [CAMELS: Catchment Attributes and MEteorology for Large-sample Studies](https://gdex.ucar.edu/dataset/camels.html).
- Set all configs in the directory *configs*.
- Change data-relevant configs in the directory *data*.
