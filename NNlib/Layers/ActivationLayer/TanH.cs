using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Layers.ActivationLayer
{
    public class TanH : ActivationLayer
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
                    outTensor.Deltas[j] = (float)(1f - (Math.Pow(inTensor.Elements[j], 2))) * inTensor.Deltas[j];
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
                    float value = (float)Math.Tanh(inTensor.Elements[j]);
                    outTensor.Elements[j] = value;
                }
            }
        }
    }
}
