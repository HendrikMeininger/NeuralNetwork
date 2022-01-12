using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NNlib.Layers
{
    public class SoftmaxLayer : Layer
    {
        private float sum;

        //inTensor = right Tensor outTensor = left Tensor, because we move from right to left (bachward)
        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];


                for (int j = 0; j < outTensor.Elements.Length; j++)
                {
                    Matrix<float> deltaMatrix = getDeltaMatrix(inTensor);
                    Matrix<float> inMatrix = getMatrix(inTensor);
                    Matrix<float> product = deltaMatrix.Transpose() * inMatrix;
                    float delta = inTensor.Elements[j] * (inTensor.Deltas[j] - product.At(0, 0));

                    outTensor.Deltas[j] = delta;
                }
            }
        }

        public void Forward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];
                float max = inTensor.Elements.Max();
                float[] normalized = inTensor.Elements.Select(x => x - max).ToArray();
                var in_exp = normalized.Select(exp);
                float[] arr = in_exp.ToArray();

                sum = in_exp.Sum();

                for (int j = 0; j < inTensor.Elements.Length; j++)
                {
                    float value = arr[j] / sum;
                    outTensor.Elements[j] = value;
                }
            }
        }


        #region private methods

        private Matrix<float> getMatrix(Tensor tensor)
        {
            return DenseMatrix.OfColumnMajor(tensor.Shape.Axis[0], tensor.Shape.Axis[1], tensor.Elements).Transpose();
        }

        private Matrix<float> getDeltaMatrix(Tensor tensor)
        {
            return DenseMatrix.OfColumnMajor(tensor.Shape.Axis[0], tensor.Shape.Axis[1], tensor.Deltas).Transpose();
        }

        private float exp(float f)
        {
            return (float) Math.Exp(f);
        }

        #endregion
    }
}
