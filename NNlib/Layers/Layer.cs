using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Layers
{
    public interface Layer
    {
        public void Forward(List<Tensor> inTensors, List<Tensor> outTensors);

        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors);

    }
}
