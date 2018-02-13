import os
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

modelPath = "./modelData"
modelName = "model.ckpt"

# print all tensors in checkpoint file
print("All")
chkp.print_tensors_in_checkpoint_file(os.path.join(modelPath, modelName), tensor_name="",
    all_tensors=True, all_tensor_names=True)

# print only tensor v1 in checkpoint file
print("\nvar1")
chkp.print_tensors_in_checkpoint_file(os.path.join(modelPath, modelName), tensor_name='var1',
    all_tensors=False, all_tensor_names=False)

# print only tensor v2 in checkpoint file
print("\nvar2")
chkp.print_tensors_in_checkpoint_file(os.path.join(modelPath, modelName), tensor_name='var2',
    all_tensors=False, all_tensor_names=False)
