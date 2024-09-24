import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradients(model, input_ts, input_meta, Z):
    input_ts.requires_grad_()  # Enable gradient tracking for input
    # model.eval()  # Set the model to evaluation mode
    output = model(
        input_ts, input_meta, Z
    ).sum()  # Forward pass (adjust for your use case)
    output.backward()  # Compute gradients
    gradients = input_ts.grad  # Get the gradients with respect to the input time series
    return gradients.abs().mean(dim=0)


def integrated_gradients(
    model, baseline_ts, baseline_meta, input_ts, input_meta, Z, steps=5
):
    # Ensure model is in evaluation mode
    model.eval()

    # Interpolation for the time series and metadata inputs
    scaled_ts_inputs_l = [
        baseline_ts + (float(i) / steps) * (input_ts - baseline_ts)
        for i in range(steps + 1)
    ]
    # Convert lists to tensors
    scaled_ts_inputs = torch.stack(scaled_ts_inputs_l)
    scaled_ts_inputs.requires_grad_()
    # Compute model predictions for each scaled input
    predictions = torch.stack(
        [model(scaled_ts_input, input_meta, Z) for scaled_ts_input in scaled_ts_inputs]
    )
    predictions.requires_grad_()
    # Compute gradients for each prediction with respect to both ts and meta inputs
    for participant in range(predictions.size(1)):
        grads_ts = [
            torch.autograd.grad(
                outputs=pred[participant, :],
                inputs=scaled_ts_input[participant, :, :],
                allow_unused=True,
                retain_graph=True,
                create_graph=True,
            )[0]
            for pred, scaled_ts_input in zip(predictions, scaled_ts_inputs)
        ]
    grads_ts = torch.stack(grads_ts).mean(dim=0)

    integrated_grads_ts = (input_ts - baseline_ts) * grads_ts

    return integrated_grads_ts


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        # Save the activations (feature maps) from the target layer during forward pass
        self.activations = output

    def __call__(self, input_ts, input_meta, input_Z):
        # Forward pass through the model
        output = self.model(input_ts, input_meta, input_Z)

        # Get the score for the target class (assuming a scalar output or a specific target class index)
        target = (
            output.mean()
        )  # Change this depending on the output (e.g., output[0][target_class])

        # Backward pass to get gradients with respect to the target
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Get the gradients from the last conv layer and apply global average pooling
        gradients = self.gradients.mean(
            dim=2, keepdim=True
        )  # Averaging over the time dimension

        # Get the feature maps from the last conv layer
        activations = self.activations

        # Compute Grad-CAM
        weights = gradients
        grad_cam = torch.sum(weights * activations, dim=1)  # Weighted sum over channels
        return torch.mean(grad_cam, dim=0)  # Return 1D CAM


def get_weights_and_biases(model):
    W = []
    B = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            W.append(module.weight.data)  # Collect weights
            if module.bias is not None:
                B.append(module.bias.data)  # Collect biases

    L = len(W)  # Number of layers with weights
    return W, B, L


def forward_with_capture(model, ts, meta, Z):
    activations = {}

    # Forward pass through convolutional layers
    ts = ts.transpose(1, 2)
    ts = model.cnn[0](ts)  # Conv1d
    activations["conv1"] = ts
    ts = model.cnn[1](ts)  # ReLU
    ts = model.cnn[2](ts)  # MaxPool1d
    activations["maxpooling"] = ts
    ts = model.cnn[3](ts)  # Conv1d
    activations["conv2"] = ts
    ts = model.cnn[4](ts)  # ReLU
    ts = model.cnn[5](ts)  # AdaptiveMaxPool1d
    activations["adaptmaxpooling"] = ts
    ts = ts.squeeze(-1)

    # Processing meta data
    meta = model.fc_meta(meta)
    activations["meta"] = meta

    # Merge and output
    combined = torch.cat((ts, meta), dim=1)
    activations["combined"] = combined
    merged = model.fc_merge(combined)
    activations["merged"] = merged
    output = model.fc_output(merged)
    activations["output"] = output + model.random_effects(Z)
    return activations


def relevance_adaptive_pooling(input, relevance):
    # Assuming relevance is pooled uniformly across the contributing inputs
    input_size = input.size(0)  # Length of sequence
    output_size = relevance.size(1)  # Reduced length after adaptive pooling
    scale_factor = input_size // output_size
    # Expand relevance back to the size of the input
    expanded_relevance = relevance.repeat_interleave(scale_factor, dim=1)
    return expanded_relevance


def relevance_linear(input, weights, relevance):
    # Implement LRP for a linear layer
    Z = torch.mm(input, weights.t()) + 1e-9  # Avoid division by zero
    S = relevance / Z
    C = torch.mm(S, weights)
    return input * C  # Element-wise relevance


def relevance_conv1d(input, weights, relevance, stride, padding):
    # Implement LRP for a convolutional layer
    Z = F.conv1d(input, weights, stride=stride, padding=padding) + 1e-9
    S = relevance / Z
    C = F.conv_transpose1d(S, weights, stride=stride, padding=padding)
    return input * C


def lrp(model, activations, R):
    R = R.clone()
    R = relevance_linear(activations["merged"], model.fc_output.weight, R)
    R = relevance_linear(activations["combined"], model.fc_merge.weight, R)
    R_ts, R_meta = R.split([model.hidden_ts_size, model.hidden_meta_size], dim=1)
    # Backpropagate relevance through metadata and time series paths
    R_ts = relevance_adaptive_pooling(activations["adaptmaxpooling"], R_ts)
    R_ts = relevance_conv1d(
        activations["conv2"], model.cnn[3].weight, R_ts, stride=1, padding=1
    )
    R_ts = relevance_conv1d(
        activations["conv1"], model.cnn[0].weight, R_ts, stride=1, padding=1
    )
    R_meta = relevance_linear(activations["meta"], model.fc_meta.weight, R_meta)

    return R_ts, R_meta
