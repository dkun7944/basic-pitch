import sys
import os

import argparse
import tensorflow as tf
import coremltools as ct
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from basic_pitch import models

def convert_keras_to_mlpackage(keras_path, mlpackage_path):
    try:
        print("Successfully imported models from basic_pitch")
    except ImportError as e:
        print(f"Error importing basic_pitch.models: {e}")
        print("Current working directory:", os.getcwd())
        print("Contents of parent directory:")
        print(os.listdir(parent_dir))
        return

    try:
        # Recreate the model architecture
        model = models.model_v2()
        
        # Load weights from the .keras file
        model.load_weights(keras_path)
        
        # Print model summary for debugging
        model.summary()
        
        # Define fixed input shape
        input_shape = tuple(1 if dim is None else dim for dim in model.input_shape)
        
        # Print input and output shapes
        print("Input shape:", model.input_shape)
        for i, output in enumerate(model.outputs):
            print(f"Output {i} shape:", output.shape)
        
        # Add input normalization
        input_name = 'input'
        output_names = [f'output_{i}' for i in range(len(model.outputs))]
        
        # Convert to Core ML with input normalization
        coreml_model = ct.convert(
            model,
            inputs=[ct.TensorType(shape=input_shape, dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT32
        )
        
        # Verify the conversion
        sample_input = np.random.rand(*input_shape).astype(np.float32)
        tf_output = model.predict(sample_input)
        coreml_output = coreml_model.predict({'input_1': sample_input})
        
        output_mapping = {
            'Identity_1': 'note',
            'Identity_2': 'onset',
            'Identity': 'contour'
        }
        
        print(f"Verifying CoreML and Tensorflow output are the same")
        for coreml_key, tf_key in output_mapping.items():
            tf_out = tf_output[tf_key]
            coreml_out = coreml_output[coreml_key]
            
            # Convert both outputs to numpy arrays with float32 dtype
            tf_out_np = np.array(tf_out, dtype=np.float32)
            coreml_out_np = np.array(coreml_out, dtype=np.float32)
            
            # Ensure shapes match
            if tf_out_np.shape != coreml_out_np.shape:
                print(f"Warning: Shape mismatch for output {tf_key}")
                print(f"TensorFlow shape: {tf_out_np.shape}, Core ML shape: {coreml_out_np.shape}")
                continue
            
            mse = np.mean((tf_out_np - coreml_out_np) ** 2)
            print(f"\nMean Squared Error for output {tf_key}: {mse}")
            if mse > 1e-6:
                print(f"Warning: Large discrepancy in output {tf_key}")
        
        # Save as MLPackage
        coreml_model.save(mlpackage_path)
        
        print(f"Conversion complete. MLPackage saved to {mlpackage_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to Core ML MLPackage")
    parser.add_argument("keras_path", type=str, help="Path to the input Keras model file (.keras)")
    parser.add_argument("mlpackage_path", type=str, help="Path to save the output MLPackage")
    args = parser.parse_args()

    convert_keras_to_mlpackage(args.keras_path, args.mlpackage_path)