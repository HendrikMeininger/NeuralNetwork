using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NNlib.Network
{
    public class SDGTrainer<T>
    {
        #region properties

        private readonly int _batchSize = 1;

        private List<T> _data;
        private List<int> _labels;

        public int AmountEpochs { get; set; }

        public bool Shuffle { get; set; } = false;

        public float LearningRate { get; set; }

        public Flavor UpdateMechanism { get; set; }

        #endregion

        #region constructor

        public SDGTrainer(List<T> trainData, List<int> labels, int amountEpochs = 10, bool shuffle = false)
        {
            _data = trainData;
            _labels = labels;
            AmountEpochs = amountEpochs;
            Shuffle = shuffle;
        }

        #endregion

        #region public methods

        public void Optimize(Network<T> network)
        {
            for (int i = 0;i < AmountEpochs;i++)
            {
                Console.WriteLine("Start Epoch " + (i + 1));
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                trainNetwork(network);
                if (Shuffle && i > 0)
                {
                    shuffleData();
                    network.LossLayer.SetExpectedValues(_labels);
                }

                stopwatch.Stop();
                Console.WriteLine("Elapsed Time is {0} s", (stopwatch.ElapsedMilliseconds / 1000));
            }
            network.LossLayer.ResetIndex();
        }

        #endregion

        #region private methods

        private void trainNetwork(Network<T> network)
        {
            List<T> list = new List<T>();
            int goodValues = 0;
            double lossSum = 0;

            for (int i = 0;i < _data.Count;i++)
            {
                if (i % 10000 == 0)
                {
                    //Console.WriteLine("Image number: " + i);
                }

                list.Clear();
                list.Add(_data[i]);
                
                if (network.Forward(list))
                {
                    goodValues++;
                }
                lossSum += network.LossLayer.GetLoss()[0];
                network.Backprop(LearningRate);
            }

            double accuracy = (double)goodValues / _data.Count;
            double loss = lossSum / _data.Count;

            Console.WriteLine("Accuracy: " + accuracy);
            Console.WriteLine("Loss: " + loss);
            network.LossLayer.ResetIndex();
        }

        private void shuffleData()
        {
            Random rn = new Random();
            int seed = rn.Next();
            ShuffleList(_data, seed);
            ShuffleList(_labels, seed);
        }

        public void ShuffleList<T>(IList<T> list, int seed)
        {
            var rng = new Random(seed);
            int n = list.Count;

            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        #endregion
    }
}
