import torch.onnx
from torch import nn

import config

import config
from dataset import BERTDataset
from model import BERTBaseUncased


if __name__ == "__main__":
    device = "cuda"
    review = ["this is an amazing video"]
    
    dataset = BERTDataset( review=review, target=[0])
    
    model = BERTBaseUncased()
    model = nn.DataParallel (model)
    model. load_state_dict(torch.load(config.MODEL_PATH))
    model . eval()
    
    print (model)
    
    ids = dataset[0]["ids"].unsqueeze(0)
    mask = dataset[0]["mask"].unsqueeze(0)
    token_type_ids = dataset[0]["token_type_ids"].unsqueeze(0)

    torch.onnx.export(
        model.module, #Need to use when Dataparallel is used
        (ids, mask, token_type_ids),
        "model.onnx",
        input_names = ["ids", "mask", "token_type_ids"] ,
        output_names=["output"],
        dynamic_axes={
            "ids" : {0: "batch _ size"},
            "mask" : {0: "batch size"},
            "token_type_ids":{0: "batch size"},
            "output ": {0: "batch size"},
        },
    )
