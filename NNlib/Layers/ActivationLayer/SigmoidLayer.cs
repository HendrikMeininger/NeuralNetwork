using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Layers.ActivationLayer
{
    public class SigmoidLayer : ActivationLayer
    {
        //inTensor = right Tensor outTensor = left Tensor, because we move from right to left (bachward)
        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                //inTensor = right Tensor outTensor = left Tensor, because we move from right to left (bachward)
                for (int j = 0; j < inTensor.Elements.Length; j++)
                {
                    //inTensor.Elements[j] == sigmoid(j) cause we calculated it earlier (in Forwardpass)
                    outTensor.Deltas[j] = inTensor.Elements[j] * (1 - inTensor.Elements[j]) * inTensor.Deltas[j];
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
                    outTensor.Elements[j] = sigmoidFunction(inTensor.Elements[j]);
                }
            }
        }

        private static float sigmoidFunction(float f)
        {
            return (float)(1 / (1 + (Math.Pow(Math.E, -f)))); ;
        }
    }
}
