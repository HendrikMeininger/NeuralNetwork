using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Layers.LossLayer
{
    public interface LossLayer
    {
        public List<float[]> ExpectedValues { get; set; }

        public void SetExpectedValues(List<int> list);

        public void Forward(List<Tensor> inTensors);

        public void Backward(List<Tensor> outTensors);

        public List<bool> GetAccuracy();

        public float[] GetLoss();

        public void ResetIndex();

    }
}
