using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NNlib.Layers.LossLayer
{
    public class MeanSquaredErrorLayer : LossLayer
    {
        public List<float[]> ExpectedValues { get; set; }

        private List<bool> _accuracy = new List<bool>();

        private float[] _loss;

        private int _index = 0;

        public MeanSquaredErrorLayer(List<float[]> expectedValues)
        {
            ExpectedValues = expectedValues;
        }

        public MeanSquaredErrorLayer(List<int> expectedValues)
        {
            SetExpectedValues(expectedValues);
        }

        public void Backward(List<Tensor> outTensors)
        {
            for (int i = 0; i < outTensors.Count; i++)
            {
                Tensor outTensor = outTensors[i];

                for (int j = 0; j < outTensor.Elements.Length; j++)
                {
                    outTensor.Deltas[j] = outTensor.Elements[j] - ExpectedValues[_index - 1][j];
                }
            }
        }

        public void Forward(List<Tensor> inTensors)
        {
            calculateAccuracy(inTensors);

            float[] res = new float[inTensors.Count];

            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                float loss = 0;

                for (int j = 0; j < inTensor.Elements.Length; j++)
                {
                    loss += (float)(0.5 * Math.Pow((inTensor.Elements[j] - ExpectedValues[_index][j]), 2));
                }

                res[i] = loss;
            }

            _index++;
            _loss = res;
        }

        public void SetExpectedValues(List<int> list)
        {
            int max = list.Max();
            ExpectedValues = new List<float[]>();
            for (int i = 0; i < list.Count; i++)
            {
                float[] arr = new float[max + 1];
                arr[list[i]] = 1;
                ExpectedValues.Add(arr);
            }
        }

        public List<bool> GetAccuracy()
        {
            return _accuracy;
        }

        private void calculateAccuracy(List<Tensor> inTensors)
        {
            _accuracy.Clear();
            float[] realLabels = ExpectedValues[_index];

            float maxValue = realLabels.Max();
            int realLabel = realLabels.ToList().IndexOf(maxValue);

            float[] predictedLabels = inTensors[0].Elements;

            float maxFloatValue = predictedLabels.Max();
            int predictedLabel = predictedLabels.ToList().IndexOf(maxFloatValue);

            _accuracy.Add(predictedLabel == realLabel);
        }
        
        public void ResetIndex()
        {
            _index = 0;
        }

        public float[] GetLoss()
        {
            return _loss;
        }
    }
}
