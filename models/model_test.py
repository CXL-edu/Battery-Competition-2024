# from model._model_basic import ModelABC
# import torch.nn as nn
#
# class ModelTest(ModelABC):
#     def __init__(self, config):
#         super(ModelTest, self).__init__(2,2,2,2)
#         self.conv = nn.Conv1d(1, 1, 1)
#
#     def forward(self, x):
#         return self.conv(x)
#
#     def get_model_info(self):
#         super().get_model_info()
#
#
# if __name__ == '__main__':
#     import torch
#     model = ModelTest({})
#     model.get_model_info()
#     x = torch.randn(1,1,1)
#     print(model(x).shape)
#     print(torch.cuda.is_available())

# class AttrDict(dict):
#     def __getattr__(self, name):
#         if name in self:
#             return self[name]
#         raise AttributeError(f"'AttrDict' object has no attribute '{name}'")
#
#     def __setattr__(self, name, value):
#         self[name] = value
#
#
# x = dict({'a':1,'b':2})
# x = AttrDict(x)
# print(x)
# x.z = 10
# print(x)
# print()

























