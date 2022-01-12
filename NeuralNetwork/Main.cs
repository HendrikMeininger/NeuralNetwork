using Keras.Datasets;
using NNlib.Extensions;
using NNlib.Layers;
using NNlib.Layers.ActivationLayer;
using NNlib.Layers.LossLayer;
using NNlib.Network;
using NNlib.Tensors;
using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Program
    {
        public static void Main(string[] args)
        {
            int trainSize = 60000;
            int testSize = 10000;

            //Network<byte[,]> network = trainNetwork(0.09f, trainSize);
            Network<byte[,]> network = trainConNetwork(0.08f, trainSize);
            //Console.WriteLine("-------------------------------------------------------------");
            testNetwork(network, testSize);
            //test();
            //debugConNetwork();
        }

        private static void test()
        {
            TensorShape inShape = new TensorShape(new int[] { 3, 3, 2 });
            TensorShape outShape = new TensorShape(new int[] { 2, 2, 2 });
            Conv2DLayer layer = new Conv2DLayer(inShape, outShape);
            layer.Bias = new Tensor(new float[] { 0, 0 }, new TensorShape(new int[] { 2, 1 }));

            float[] elements1 = new float[] { 1, 3, 2, 4, 2, 3, 2, 2, 1, 1, 2, 2, 3, 4, 3, 2, 2, 1 };
            Tensor tensor1 = new Tensor(elements1, inShape);

            float[] elements2 = new float[2 * 2 * 2 * 2];
            Tensor tensor2 = new Tensor(elements2, outShape);

            layer.Filter = new Tensor(new float[] { 2, 1, 4, 3, 1, 1, 3, 2, 3, 1, 4, 1, 2, 1, 1, 3 }, new TensorShape(new int[] { 2, 2, 2, 2 }));
            float[] elements3 = new float[] { 4, 3, 5, 2, 1, 6, 2, 3 };
            tensor2.Deltas = elements3;

            layer.calculateDeltaWeights(new List<Tensor>() { tensor2 }, new List<Tensor>() { tensor1 }, 0);

        }

        #region train network

        private static Network<byte[,]> trainNetwork(float learningRate, int trainSize)
        {
            List<int> labels = new List<int>();
            List<byte[,]> data = new List<byte[,]>();

            MNIST.GetTrainingSet(data, labels, trainSize);

            TensorShape firstShape = new TensorShape(new int[] { 1, 784 });
            TensorShape secondShape = new TensorShape(new int[] { 1, 100 });
            TensorShape thirdShape = new TensorShape(new int[] { 1, 10 });

            InputLayer<byte[,]> inputLayer = new InputLayer<byte[,]>();
            List<Layer> layers = new List<Layer>();

            layers.Add(new FullyConnectedLayer(firstShape, secondShape));
            layers.Add(new SigmoidLayer());
            layers.Add(new FullyConnectedLayer(secondShape, thirdShape));
            layers.Add(new SoftmaxLayer());

            LossLayer lossLayer = new CrossEntropyLayer(labels);
            Network<byte[,]> network = new Network<byte[,]>(inputLayer, layers, lossLayer, 1, new TensorShape(new int[] { 1, 784 }));

            SDGTrainer<byte[,]> trainer = new SDGTrainer<byte[,]>(data, labels, amountEpochs:20, shuffle:true);
            trainer.LearningRate = learningRate;
            trainer.Optimize(network);

            return network;
        }

        private static Network<byte[,]> trainConNetwork(float learningRate, int trainSize)
        {
            List<int> labels = new List<int>();
            List<byte[,]> data = new List<byte[,]>();

            MNIST.GetTrainingSet(data, labels, trainSize);

            int filterNumber = 5;

            TensorShape firstShape = new TensorShape(new int[] { 1, 784 }); 
            TensorShape secondShape = new TensorShape(new int[] { 1, 400});
            TensorShape thirdShape = new TensorShape(new int[] { 20, 20, 1});
            TensorShape fourthShape = new TensorShape(new int[] { 16, 16, filterNumber });
            TensorShape fifthShape = new TensorShape(new int[] { 1, filterNumber * 256 });
            TensorShape sixthShape = new TensorShape(new int[] { 1, 10 });

            InputLayer<byte[,]> inputLayer = new InputLayer<byte[,]>();
            List<Layer> layers = new List<Layer>();

            layers.Add(new FullyConnectedLayer(firstShape, secondShape));
            layers.Add(new SigmoidLayer());
            layers.Add(new FlattenLayer(secondShape, thirdShape));
            layers.Add(new Conv2DLayer(thirdShape, fourthShape));
            layers.Add(new SigmoidLayer());
            layers.Add(new FlattenLayer(fourthShape, fifthShape));
            layers.Add(new FullyConnectedLayer(fifthShape, sixthShape));
            layers.Add(new SoftmaxLayer());
             
            LossLayer lossLayer = new CrossEntropyLayer(labels);
            Network<byte[,]> network = new Network<byte[,]>(inputLayer, layers, lossLayer, 1, new TensorShape(new int[] { 1, 784 }));

            SDGTrainer<byte[,]> trainer = new SDGTrainer<byte[,]>(data, labels, amountEpochs:10, shuffle:true);
            trainer.LearningRate = learningRate;
            trainer.Optimize(network);

            return network;
        }

        #endregion

        #region test network

        private static void testNetwork(Network<byte[,]> network, int testSize)
        {
            List<int> labels = new List<int>();
            List<byte[,]> data = new List<byte[,]>();

            MNIST.GetTestSet(data, labels, testSize);

            network.Test(data, labels);
        }

        #endregion

        #region debug networks

        private static void debugConNetwork()
        {
            TensorShape inShape = new TensorShape(new int[] { 4, 3, 2 });
            TensorShape outShape = new TensorShape(new int[] { 3, 2, 2 });
            Conv2DLayer layer = new Conv2DLayer(inShape, outShape);

            float[] elements1 = new float[] { 0.1f, 1.2f, 0.01f, -0.2f, 1.4f, 0.2f, 0.5f, 1.6f, -0.3f, 0.6f, 2.2f, 4.0f,
                0.9f, 1.1f, 3.2f, 0.3f, 0.7f, 1.7f, 0.5f, 2.2f, 6.3f, 0.65f, 4.4f, 8.2f };
            Tensor tensor1 = new Tensor(elements1, inShape);

            float[] elements2 = new float[3 * 2 * 2];
            Tensor tensor2 = new Tensor(elements2, outShape);

            layer.Filter = new Tensor(new float[] { 0.1f, 0.3f,
                -0.2f, 0.4f,
                0.7f, 0.9f,
                0.6f, -1.1f,
                0.37f, 0.32f,
                -0.9f, 0.17f,
                0.9f, 0.2f,
                0.3f, -0.7f}, new TensorShape(new int[] { 2, 2, 2, 2 }));
            layer.Forward(new List<Tensor>() { tensor1 }, new List<Tensor>() { tensor2 });


            float[] elements3 = new float[] { 0.1f, -0.25f, 0.33f, 1.3f, -0.6f, 0.01f, -0.5f, -0.8f, 0.2f, 0.81f, 0.1f, 1.1f };
            tensor2.Deltas = elements3;

            layer.Backward(new List<Tensor>() { tensor2 }, new List<Tensor>() { tensor1 });

            layer.calculateDeltaWeights(new List<Tensor>() { tensor2 }, new List<Tensor>() { tensor1 }, 0);
        }

        private static void debugNetwork()
        {
            float[] label = new float[] { 0.7095f, 0.0942f };
            List<float[]> labels = new List<float[]>();
            labels.Add(label);

            TensorShape inShape = new TensorShape(new int[] { 1, 3 });
            float[] inElements = new float[] { 0.4183f, 0.5209f, 0.0291f };
            Tensor inTensor = new Tensor(inElements, inShape);

            TensorShape tensorShape1 = new TensorShape(new int[] { 3, 3 });
            float[] elements1 = new float[] { -0.5057f, 0.3987f, -0.8943f, 0.3356f, 0.1673f, 0.8321f, -0.3485f, -0.4597f, -0.1121f };
            Tensor tensor1 = new Tensor(elements1, tensorShape1);

            TensorShape tensorShape2 = new TensorShape(new int[] { 3, 2 });
            float[] elements2 = new float[] { 0.4047f, 0.9563f, -0.8192f, -0.1274f, 0.3662f, -0.7252f };
            Tensor tensor2 = new Tensor(elements2, tensorShape2);

            TensorShape firstShape = new TensorShape(new int[] { 1, 3 });
            TensorShape secondShape = new TensorShape(new int[] { 1, 3 });
            FullyConnectedLayer fullyConnectedLayer1 = new FullyConnectedLayer(firstShape, secondShape);
            fullyConnectedLayer1.WeightTensor = tensor1;

            SigmoidLayer sigmoidLayer = new SigmoidLayer();

            TensorShape thirdShape = new TensorShape(new int[] { 1, 2 });
            FullyConnectedLayer fullyConnectedLayer2 = new FullyConnectedLayer(secondShape, thirdShape);
            fullyConnectedLayer2.WeightTensor = tensor2;

            SoftmaxLayer softmaxLayer = new SoftmaxLayer();
            LossLayer lossLayer = new MeanSquaredErrorLayer(labels);

            Tensor outTensor1 = new Tensor(new float[3], firstShape);
            fullyConnectedLayer1.Forward(new List<Tensor>() { inTensor }, new List<Tensor>() { outTensor1 });

            Tensor outTensor2 = new Tensor(new float[3], secondShape);
            sigmoidLayer.Forward(new List<Tensor>() { outTensor1 }, new List<Tensor>() { outTensor2 });

            Tensor outTensor3 = new Tensor(new float[2], thirdShape);
            fullyConnectedLayer2.Forward(new List<Tensor>() { outTensor2 }, new List<Tensor>() { outTensor3 });

            Tensor outTensor4 = new Tensor(new float[2], thirdShape);
            softmaxLayer.Forward(new List<Tensor>() { outTensor3 }, new List<Tensor>() { outTensor4 });

            //lossLayer.Forward(new List<Tensor>() { outTensor4 });
            //lossLayer.Backward(new List<Tensor>() { outTensor4 });

            outTensor4.Deltas = new float[] { -1.4901f, -0.1798f };

            softmaxLayer.Backward(new List<Tensor>() { outTensor4 }, new List<Tensor>() { outTensor3 });
            //outTensor3.Deltas = new float[] {-1.4901, -0.1798}

            fullyConnectedLayer2.Backward(new List<Tensor>() { outTensor3 }, new List<Tensor>() { outTensor2 });
            fullyConnectedLayer2.calculateDeltaWeights(new List<Tensor>() { outTensor3 }, new List<Tensor>() { outTensor2 }, 1);

            sigmoidLayer.Backward(new List<Tensor>() { outTensor2 }, new List<Tensor>() { outTensor1 });


            fullyConnectedLayer1.calculateDeltaWeights(new List<Tensor>() { outTensor1 }, new List<Tensor>() { inTensor }, 1);
        }

        #endregion
    }
}

