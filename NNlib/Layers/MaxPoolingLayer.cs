using NNlib.Extensions;
using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Layers
{
    public class MaxPoolingLayer : Layer
    {
        #region private variables

        private TensorShape _pooling2DShape;

        private List<List<int[]>> _masks;

        public TensorShape InShape { get; set; }

        public TensorShape OutShape { get; set; }

        #endregion

        #region constructor

        public MaxPoolingLayer(TensorShape inShape, TensorShape outShape, TensorShape pooling2DShape)
        {
            _pooling2DShape = pooling2DShape;
            if (pooling2DShape.Axis[0] != pooling2DShape.Axis[1])
            {
                throw new NotImplementedException();
            }

            _masks = new List<List<int[]>>();
            InShape = inShape;
            OutShape = outShape;
        }

        #endregion

        #region implement interface

        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0;i < inTensors.Count;i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                float[] newDeltas = new float[outTensor.Deltas.Length];
                List<int[]> masks = _masks[i];

                int channelSize = outTensor.Shape.Axis[0] * outTensor.Shape.Axis[1];

                for (int j = 0; j < inTensor.Shape.Axis[2];j++)
                {
                    int[] mask = masks[j];

                    for (int k = 0;k < mask.Length;k++)
                    {
                        int index = mask[k];
                        newDeltas[index] = inTensor.Deltas[k];
                    }
                }
                outTensor.Deltas = newDeltas;
            }
        }

        public void Forward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            int poolSize = _pooling2DShape.Axis[0] * _pooling2DShape.Axis[1];
            _masks.Clear();

            for (int i = 0;i < inTensors.Count;i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                List<int[]> masks = new List<int[]>();

                int stride = _pooling2DShape.Axis[0];
                int inChannelSize = inTensor.Shape.Axis[0] * inTensor.Shape.Axis[1];
                int outChannelSize = outTensor.Shape.Axis[0] * outTensor.Shape.Axis[1];

                for (int j = 0; j < inTensor.Shape.Axis[2];j++)
                {
                    int[] mask = new int[outTensor.Shape.Axis[0] * outTensor.Shape.Axis[1]];
                    float maxVal = float.MinValue;
                    int maxValPosition = -1;
                    int index = 0;

                    for (int k = 0; k < outTensor.Shape.Axis[0] * outTensor.Shape.Axis[1]; k++)
                    {
                        if (index % inTensor.Shape.Axis[1] == 0 && index != 0)
                        {
                            index += inTensor.Shape.Axis[1];
                        }

                        for (int m = 0;m < stride;m++)
                        {
                            for (int n = 0; n < stride; n++)
                            {
                                int position = j * inChannelSize + index + m * inTensor.Shape.Axis[0] + n;
                                float val = inTensor.Elements[position];

                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxValPosition = position;
                                }
                            }
                        }

                        mask[k] = maxValPosition;
                        outTensor.Elements[j * outChannelSize + k] = maxVal;

                        index += 2;
                    }

                    masks.Add(mask);
                }

                _masks.Add(masks);
            }
        }

        #endregion
    }
}
