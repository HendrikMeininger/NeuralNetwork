using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public static class MNIST
    {
        #region public methods

        public static void GetTrainingSet(List<byte[,]> data, List<int> labels, int imageNumber = 60000)
        {
            if (imageNumber > 60000)
            {
                throw new NotImplementedException("The Dataset only has 60000 entries");
            }

            List<DigitImage> trainData = readMNISTTrain(imageNumber);

            foreach (DigitImage image in trainData)
            {
                labels.Add(image.label);
                data.Add(To2D(image.pixels));
            }
        }

        public static void GetTestSet(List<byte[,]> data, List<int> labels, int imageNumber = 10000)
        {
            if (imageNumber > 10000)
            {
                throw new NotImplementedException("The Dataset only has 10000 entries");
            }

            List<DigitImage> trainData = readMNISTTest(imageNumber);

            foreach (DigitImage image in trainData)
            {
                labels.Add(image.label);
                data.Add(To2D(image.pixels));
            }
        }

        #endregion

        #region private methods

        private static List<DigitImage> readMNISTTrain(int imageNumber)
        {
            string labelPath = @"C:\train-labels.idx1-ubyte";
            string imagePath = @"C:\train-images.idx3-ubyte";

            return loadFromFile(labelPath, imagePath, imageNumber);
        }

        private static List<DigitImage> readMNISTTest(int imageNumber)
        {
            string labelPath = @"C:\t10k-labels.idx1-ubyte";
            string imagePath = @"C:\t10k-images.idx3-ubyte";

            return loadFromFile(labelPath, imagePath, imageNumber);
        }

        private static List<DigitImage> loadFromFile(string labelPath, string imagePath, int imageCount)
        {
            FileStream ifsLabels =
             new FileStream(labelPath,
             FileMode.Open); // test labels
            FileStream ifsImages =
             new FileStream(imagePath,
             FileMode.Open); // test images

            BinaryReader brLabels =
             new BinaryReader(ifsLabels);
            BinaryReader brImages =
             new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];

            List<DigitImage> list = new List<DigitImage>();

            // each test image
            for (int di = 0; di < imageCount; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }

                byte lbl = brLabels.ReadByte();

                DigitImage dImage =
                  new DigitImage(pixels, lbl);

                list.Add(dImage);

            }

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            return list;
        }

        private static T[,] To2D<T>(T[][] source)
        {
            try
            {
                int FirstDim = source.Length;
                int SecondDim = source.GroupBy(row => row.Length).Single().Key; // throws InvalidOperationException if source is not rectangular

                var result = new T[FirstDim, SecondDim];
                for (int i = 0; i < FirstDim; ++i)
                    for (int j = 0; j < SecondDim; ++j)
                        result[i, j] = source[i][j];

                return result;
            }
            catch (InvalidOperationException)
            {
                throw new InvalidOperationException("The given jagged array is not rectangular.");
            }
        }

        private class DigitImage
        {
            public byte[][] pixels;
            public byte label;

            public DigitImage(byte[][] pixels,
              byte label)
            {
                this.pixels = new byte[28][];
                for (int i = 0; i < this.pixels.Length; ++i)
                    this.pixels[i] = new byte[28];

                for (int i = 0; i < 28; ++i)
                    for (int j = 0; j < 28; ++j)
                        this.pixels[i][j] = pixels[i][j];

                this.label = label;
            }

            public override string ToString()
            {
                string s = "";
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        if (this.pixels[i][j] == 0)
                            s += " "; // white
                        else if (this.pixels[i][j] == 255)
                            s += "X"; // black
                        else
                            s += "#"; // gray
                    }
                    s += "\n";
                }
                s += this.label.ToString();
                return s;


            }
        }

        #endregion
    }
}
