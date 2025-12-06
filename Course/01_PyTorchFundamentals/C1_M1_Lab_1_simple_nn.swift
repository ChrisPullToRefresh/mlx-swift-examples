// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

@main
struct Train {
    
    static func main() {
        print("Starting training...")
        
        // Set random seed for reproducibility (equivalent to torch.manual_seed(42))
        MLXRandom.seed(42)
        
        let epochs = 1000
        let batchSize = 4
        let compile = true
        let device = DeviceType.cpu
        
        #if compiler(>=6.0)
        #warning("Device.setDefault is deprecated but withDefaultDevice is not yet available in MLX")
        #endif
        Device.setDefault(device: Device(device))
        print("Device set to: \(device)")

        // A very simple model using Sequential and Linear layers
        // This implements a linear function: y = mx + b
        let model = Sequential(layers: Linear(inputDimensions: 1, outputDimensions: 1))
        print("Model created")
        eval(model.parameters())
        print("Model parameters evaluated")

        // Measure the distance from the prediction (model(x)) and the
        // ground truth (y). This gives feedback on how close the
        // prediction is from matching the truth.
        func loss(model: Sequential, x: MLXArray, y: MLXArray) -> MLXArray {
            mseLoss(predictions: model(x), targets: y, reduction: .mean)
        }

        let lg = valueAndGrad(model: model, loss)
        print("valueAndGrad created")

        // The optimizer will use the gradients update the model parameters
        let optimizer = SGD(learningRate: 0.01)

        func step(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
            let (loss, grads) = lg(model, x, y)
            optimizer.update(model: model, gradients: grads)
            return loss
        }

        let resolvedStep =
            compile
            ? MLX.compile(inputs: [model, optimizer], outputs: [model, optimizer], step) : step
        let rawDistances: [Float32] = [1.0, 2.0, 3.0, 4.0]
        let distances = MLXArray(rawDistances, [batchSize, 1])
        let rawTimes: [Float32] = [6.96, 12.11, 16.77, 22.21]
        let times = MLXArray(rawTimes, [batchSize, 1])
        print("Training data created: distances=\(distances.shape), times=\(times.shape)")
        
        for epoch in 0 ..< epochs {
            
            
            // Notice that the shape is [B, 1] where B is the batch
            // dimension. This allows us to train on several samples simultaneously.
            //
            // Note: A very large batch size will take longer to converge because
            // the gradient will be representing too many samples down into
            // a single float parameter.

            // Compute the loss and gradients. Use the optimizer
            // to adjust the parameters closer to the target.
            let loss = resolvedStep(distances, times)

            eval(model, optimizer)

            // We should see this converge toward 0
            if (epoch+1).isMultiple(of: 100) {
                print("\n=== Epoch \(epoch + 1)/\(epochs) ===")
                print("parameters: \(model.parameters())")
                print("loss: \(loss)")
            }
        }
        
        print("\nTraining complete!")

    }
}
