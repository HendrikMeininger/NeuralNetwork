using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Layers.ActivationLayer
{
    public class Relu : ActivationLayer
    {
        //inTensor = right Tensor outTensor = left Tensor, because we move from right to left (bachward)
        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                for (int j = 0; j < inTensor.Elements.Length; j++)
                {
                    if (inTensor.Elements[j] < 0)
                    {
                        outTensor.Deltas[j] = 0;
                    }
                    else
                    {
                        outTensor.Deltas[j] = inTensor.Deltas[j];
                    }
                }
            }
        }

        public void Forward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                for (int j = 0; j < inTensor.Elements.Length; j++)
                {
                    if (inTensor.Elements[j] < 0)
                    {
                        outTensor.Elements[j] = 0;
                    }
                    else
                    {
                        outTensor.Elements[j] = inTensor.Elements[j];
                    }
                }
            }
        }
    }
}
