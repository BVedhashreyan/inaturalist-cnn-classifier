import wandb
from train import main

sweep_config = {
    "method": "bayes",

    "metric": {
        "name": "best_val_acc",
        "goal": "maximize"
    },

    "parameters": {

        "activation": {
            "values": ["relu", "gelu", "silu"]
        },

        "dropout": {
            "values": [0.0, 0.2, 0.3]
        },

        "batch_norm": {
            "values": [True, False]
        },

        "filters": {
            "values": [
                [32,32,32,32,32],
                [32,64,128,64,32],
                [32,64,128,256,512]
            ]
        },

        "kernel_size": {
            "values": [3, 5]
        },  

        "learning_rate": {
            "values": [1e-3, 5e-4]
        },

        "batch_size": {
            "values": [32, 64]
        },

        "epochs": {
            "value": 5
        }
    }
}

sweep_id = wandb.sweep(
    sweep_config,
    project="inaturalist-cnn-classifier"
)

wandb.agent(
    sweep_id,
    function=main,
    count=3
)