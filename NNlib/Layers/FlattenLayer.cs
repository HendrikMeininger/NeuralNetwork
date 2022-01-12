using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Layers
{
    public class FlattenLayer : Layer
    {
        #region properties

        public TensorShape InShape { get; set; }

        public TensorShape OutShape { get; set; }

        #endregion

        #region constructor

        public FlattenLayer(TensorShape inShape, TensorShape outShape)
        {
            InShape = inShape;
            OutShape = outShape;
        }

        #endregion

        #region interface methods implementation

        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count;i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                outTensor.Deltas = inTensor.Deltas;
                outTensor.Elements = inTensor.Elements;
                outTensor.Shape = InShape;
            }
        }

        public void Forward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                outTensor.Deltas = inTensor.Deltas;
                outTensor.Elements = inTensor.Elements;
                outTensor.Shape = OutShape;
            }
        }

        #endregion
    }
}
