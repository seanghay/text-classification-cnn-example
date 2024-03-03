from models import TextCNN
import torch

model = TextCNN()
model.load_state_dict(torch.load("checkpoint_14.pth"))
model.eval()
# Set the input shape (replace with your actual input size)

with torch.no_grad():
    batch_size = 1  # Example batch size
    max_seq_len = 128  # Example maximum sequence length
    input_ids = torch.randint(low=0, high=8000, size=(batch_size, max_seq_len))
    print(input_ids.shape)

    # Export the model
    torch.onnx.export(
        model,  # Model to export
        (input_ids,),  # Dummy input for shape definition
        "text_cnn.onnx",  # Output filename
        verbose=False,  # Optional: Print information during export
        opset_version=14,  # Optional: Specify ONNX opset version (>= 10 for most operators)
        export_params=True,
        do_constant_folding=True,
        output_names=["output"],
        input_names=["input"],
        dynamic_axes={"input": {1: "sequence_length"}},
    )

    print("Model exported successfully to text_cnn.onnx")
