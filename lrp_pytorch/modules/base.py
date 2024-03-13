import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def safe_divide(numerator, divisor, eps0, eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign())


def lrp_backward(input_, layer, relevance_output, eps0, eps):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)
    Z.backward(S)
    relevance_input = input_.data * input_.grad
    layer.saved_relevance = relevance_input
    return relevance_input


def cam_backward(input_, layer, relevance_output):
    if input_.grad is not None:
        input_.grad.zero()
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(input_)
    Z.backward(relevance_output_data)
    relevance_input = F.relu(input_.data * input_.grad)
    layer.saved_relevance = relevance_input
    return relevance_input


def eb_backward(input_, layer, relevance_output):
    layer: nn.Linear
    with torch.no_grad():
        try:
            layer.weight.copy_(F.relu(layer.weight))
        except:
            pass
    if input_.grad is not None:
        input_.grad.zero()
    with torch.enable_grad():
        Z = layer(input_)  # X = W^{+}^T * A_{n}
    relevance_output_data = relevance_output.clone().detach()  # P_{n-1}
    X = Z.clone().detach()
    Y = relevance_output_data / X  # Y = P_{n-1} (/) X
    Z.backward(Y)  # Use backward pass to compute Z = W^{+} * Y
    relevance_input = input_.data * input_.grad  # P_{n} = A_{n} (*) Z
    layer.saved_relevance = relevance_input
    return relevance_input


def setbyname(obj, name, value):
    # print(obj, name, value)

    def iteratset(obj, components, value):
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            # print('set!!', components[0])
            # exit()
            return True
        else:
            nextobj = getattr(obj, components[0])
            # print(nextobj, components)
            return iteratset(nextobj, components[1:], value)

    components = name.split('.')
    success = iteratset(obj, components, value)
    return success


def getbyname(obj, name):
    # print(obj, name, value)

    def iteratget(obj, components):
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            value = getattr(obj, components[0])
            print('found!!', components[0])
            # exit()
            return value
        else:
            nextobj = getattr(obj, components[0])
            # print(nextobj, components)
            return iteratget(nextobj, components[1:])

    components = name.split('.')
    success = iteratget(obj, components)
    return success


# TODO: Finish this
# def convert_model(model: nn.Module, lrp_params):
#     model_copy = deepcopy(model)
#     copied_module_names = []
#     for name, module in model_copy.named_modules():
#         lrp_module = get_lrpwrappermodule(module, lrp_params)
#         if lrp_module is None:
#             print('No corresponding LRP module', name, module)
#         elif isinstance(module, type(model_copy)):
#             print('Skipping, Main object')
#         else:
#             print(f'Setting {name}', module)
#             success = setbyname(model_copy, name, lrp_module)
#             assert success
#             if success:
#                 copied_module_names.append(name)
#             else:
#                 print(f'Failed to set {name}', module, lrp_module)
#     print(copied_module_names)
#     for name, module in model_copy.named_modules():
#         if name not in copied_module_names:
#             print(f'Not wrapped: {name}')
#     return model_copy
