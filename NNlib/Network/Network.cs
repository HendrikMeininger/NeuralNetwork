using NNlib.Layers;
using NNlib.Layers.LossLayer;
using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Network
{
    public class Network<T>
    {
        public InputLayer<T> Input { get; set; }

        public List<Layer> Layers { get; set; }

        public LossLayer LossLayer { get; set; }

        public List<Tensor> Parameters { get; set; }

        public List<Tensor> DeltaParams { get; set; }


        private List<List<Tensor>> _tensors = new List<List<Tensor>>();


        #region constructor

        public Network(InputLayer<T> inputLayer, List<Layer> layers, LossLayer lossLayer, int batchSize, TensorShape shape)
        {
            Input = inputLayer;
            Layers = layers;
            LossLayer = lossLayer;

            createTensors(batchSize, shape);
        }

        #endregion

        #region public methods

        public float Test(List<T> testData, List<int> labels)
        {
            float loss = 0;
            int goodValues = 0;

            LossLayer.SetExpectedValues(labels);
            for (int i = 0; i < testData.Count; i++)
            {
                List<T> list = new List<T>();
                list.Add(testData[i]);

                if (Forward(list))
                {
                    goodValues++;
                }
            }

            double accuracy = (double)goodValues / testData.Count;
            Console.WriteLine("Accuracy: " + accuracy);
            LossLayer.ResetIndex();

            return 0;
        }

        public void Backprop(float learningRate)
        {
            LossLayer.Backward(_tensors[_tensors.Count - 1]);
            for (int i = Layers.Count; i > 0; i--)
            {
                Layer layer = Layers[i - 1];
                layer.Backward(_tensors[i], _tensors[i - 1]);
                if (layer is FullyConnectedLayer)
                {
                    ((FullyConnectedLayer)layer).calculateDeltaWeights(_tensors[i], _tensors[i - 1], learningRate);
                }
                if (layer is Conv2DLayer)
                {
                    ((Conv2DLayer)layer).calculateDeltaWeights(_tensors[i], _tensors[i - 1], learningRate);
                }
            }
        }

        public bool Forward(List<T> data)
        {
            List<Tensor> inTensor = Input.Forward(data);
            _tensors[0] = inTensor;

            for (int i = 0;i < Layers.Count;i++)
            {
                
                Layer layer = Layers[i];
                if (i == 0)
                {
                    layer.Forward(inTensor, _tensors[i + 1]);
                }
                else
                {
                    layer.Forward(_tensors[i], _tensors[i + 1]);
                }
            }

            LossLayer.Forward(_tensors[_tensors.Count - 1]);

            return LossLayer.GetAccuracy()[0];
        }

        #endregion

        #region private methods

        private void createTensors(int batchSize, TensorShape firstShape)
        {
            List<Tensor> list = new List<Tensor>();
            int elementNumber;
            if (firstShape.Axis.Length > 2)
            {
                elementNumber = firstShape.Axis[0] * firstShape.Axis[1] * firstShape.Axis[2];
            }
            else
            {
                elementNumber = firstShape.Axis[0] * firstShape.Axis[1];
            }

            float[] elements = new float[elementNumber];
            Tensor t = new Tensor(elements, firstShape);
            list.Add(t);
            _tensors.Add(list);

            for (int i = 0; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                list = new List<Tensor>();

                for (int j = 0; j < batchSize; j++)
                {
                    TensorShape shape;
                    if (layer is FullyConnectedLayer)
                    {
                        shape = ((FullyConnectedLayer)layer).OutShape;
                    }
                    else if (layer is Conv2DLayer)
                    {
                        shape = ((Conv2DLayer)layer).OutShape;
                    }
                    else if (layer is FlattenLayer)
                    {
                        shape = ((FlattenLayer)layer).OutShape;
                    }
                    else if (layer is MaxPoolingLayer)
                    {
                        shape = ((MaxPoolingLayer)layer).OutShape;
                    }
                    else if (i == 0)
                    {
                        shape = firstShape;
                    }
                    else
                    {
                        shape = _tensors[i][0].Shape;
                    }
                    if (shape.Axis.Length > 2)
                    {
                        elementNumber = shape.Axis[0] * shape.Axis[1] * shape.Axis[2];
                    }
                    else
                    {
                        elementNumber = shape.Axis[0] * shape.Axis[1];
                    }
                    elements = new float[elementNumber];
                    t = new Tensor(elements, shape);
                    list.Add(t);
                }
                _tensors.Add(list);
            }
        }

        #endregion
    }
}
