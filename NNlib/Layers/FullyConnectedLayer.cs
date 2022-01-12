using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Distributions;

namespace NNlib.Layers
{
    public class FullyConnectedLayer : Layer
    {
        #region propertiies

        public Tensor WeightTensor { get; set; }

        public Tensor Bias { get; set; }

        public TensorShape InShape { get; set; }

        public TensorShape OutShape { get; set; }

        #endregion

        #region constructor

        public FullyConnectedLayer(TensorShape inShape, TensorShape outShape)
        {
            InShape = inShape;
            OutShape = outShape;

            if (inShape.Axis.Length > 2 || outShape.Axis.Length > 2)
            {
                throw new NotImplementedException();
            }

            initBias();
            initWeights();
        }

        private void initBias()
        {
            float[] elements = new float[OutShape.Axis[0] * OutShape.Axis[1]];
            Bias = new Tensor(elements, OutShape);
        }

        private void initWeights()
        {
            double mean = 0;
            int inSize = InShape.Axis[0] * InShape.Axis[1];
            int outSize = OutShape.Axis[0] * OutShape.Axis[1];

            double val = 2f / (inSize + outSize);
            double stdDev = Math.Sqrt(val);

            Normal normalDist = new Normal(mean, stdDev);

            TensorShape shape = new TensorShape(new int[] { InShape.Axis[1], OutShape.Axis[1] });
            float[] elements = new float[InShape.Axis[1] * OutShape.Axis[1]];
            Random rn = new Random();

            for (int i = 0;i < elements.Length;i++)
            {
                elements[i] = (float)rn.NextDouble() * 2 - 1;
                //elements[i] = (float)normalDist.Sample();
                //elements[i] = 0;
            }

            WeightTensor = new Tensor(elements, shape);
        }

        #endregion

        #region interface methods implementation

        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                Matrix<float> weightMatrix = getTransposedMatrix(WeightTensor);
                Matrix<float> inDeltaMatrix = getDeltaMatrix(inTensor);

                Matrix<float> newMatrix = inDeltaMatrix * weightMatrix;
                outTensor.Deltas = newMatrix.Transpose().ToColumnMajorArray();
            }
        }

        public void calculateDeltaWeights(List<Tensor> inTensors, List<Tensor> outTensors, float learningRate)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                float[] biasArr = Bias.Elements.Zip(inTensors[i].Deltas, (x, y) => x - y * learningRate).ToArray();
                Bias.Elements = biasArr;

                Matrix<float> transposedOutMatrix = getTransposedMatrix(outTensors[i]);
                Matrix<float> inDeltaMatrix = getDeltaMatrix(inTensors[i]);
                Matrix<float> newMatrix = transposedOutMatrix * inDeltaMatrix;

                Matrix<float> weightMatrix = getMatrix(WeightTensor);
                WeightTensor.Elements = (weightMatrix - newMatrix.Multiply(learningRate)).ToRowMajorArray();
            }
        }

        public void Forward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                Matrix<float> weightMatrix = getMatrix(WeightTensor);
                Matrix<float> inMatrix = getMatrix(inTensor);
                Matrix<float> biasMatrix = getMatrix(Bias);

                Matrix<float> newMatrix = inMatrix * weightMatrix + biasMatrix;

                outTensor.Elements = newMatrix.Transpose().ToColumnMajorArray();
            }
        }

        #endregion

        #region private methods

        private Matrix<float> getMatrix(Tensor tensor)
        {
            return new DenseMatrix(tensor.Shape.Axis[1], tensor.Shape.Axis[0], tensor.Elements).Transpose();
        }

        private Matrix<float> getTransposedMatrix(Tensor tensor)
        {
            return new DenseMatrix(tensor.Shape.Axis[1], tensor.Shape.Axis[0], tensor.Elements);
        }

        private Matrix<float> getDeltaMatrix(Tensor tensor)
        {
            return new DenseMatrix(tensor.Shape.Axis[1], tensor.Shape.Axis[0], tensor.Deltas).Transpose();
        }

        #endregion
    }
}
