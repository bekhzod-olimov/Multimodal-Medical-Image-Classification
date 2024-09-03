import timm, torch
from torch.nn import *

sigmoid = Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(Module):
    def forward(self, x): return Swish.apply(x)

class CustomModel(Module):
    def __init__(self, model_name, n_cls, n_features = 0, n_feature_dims = [512, 128]):
        super().__init__()
        self.n_features = n_features
        self.model = timm.create_model(model_name, pretrained = True)
        in_chs = self.model.head.fc.in_features
        self.avgpool = AvgPool2d(7)
        if n_features > 0:
            self.meta_data_feature_extractor = Sequential(
                Linear(n_features, n_feature_dims[0]),
                BatchNorm1d(n_feature_dims[0]),
                Swish_Module(),
                Dropout(p=0.3),
                Linear(n_feature_dims[0], n_feature_dims[1]),
                BatchNorm1d(n_feature_dims[1]),
                Swish_Module(),
            )
            
            in_chs += n_feature_dims[1]
        self.classifier = Linear(in_chs, n_cls)
        
    def forward(self, inp, inp_meta=None):
        
        fts = self.avgpool(self.model.forward_features(inp)).squeeze(-1).squeeze(-1)
        
        if inp_meta != None: fts = torch.cat([fts, self.meta_data_feature_extractor(inp_meta)], dim = 1)
            
        return self.classifier(fts)