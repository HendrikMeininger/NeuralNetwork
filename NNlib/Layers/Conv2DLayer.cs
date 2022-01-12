using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using NNlib.Enums;
using NNlib.Extensions;
using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NNlib.Layers
{
    public class Conv2DLayer : Layer
    {
        public Padding Padding { get; set; }

        public Tensor Filter { get; set; }

        public Tensor Bias { get; set; }

        public TensorShape InShape { get; set; }

        public TensorShape OutShape { get; set; }

        #region constructor

        public Conv2DLayer(TensorShape inShape, TensorShape outShape)
        {
            InShape = inShape;
            OutShape = outShape;

            if (inShape.Axis.Length != 3)
            {
                throw new NotImplementedException();
            }

            initBias();
            initFilter();
        }

        private void initBias()
        {
            float[] elements = new float[OutShape.Axis[2]];
            Bias = new Tensor(elements, new TensorShape(new int[] { 1, OutShape.Axis[2]}));
        }

        private void initFilter()
        {
            double mean = 0;
            int inSize = InShape.Axis[0] * InShape.Axis[1];
            int outSize = OutShape.Axis[0] * OutShape.Axis[1];

            double val = 2f / (inSize + outSize);
            double stdDev = Math.Sqrt(val);

            Normal normalDist = new Normal(mean, stdDev);

            int filterX = InShape.Axis[0] + 1 - OutShape.Axis[0];
            int filterY = InShape.Axis[1] + 1 - OutShape.Axis[1];

            TensorShape shape = new TensorShape(new int[] { filterX, filterY, InShape.Axis[2], OutShape.Axis[2] });
            float[] elements = new float[filterX * filterY * InShape.Axis[2] * OutShape.Axis[2]];
            Random rn = new Random();

            for (int i = 0; i < elements.Length; i++)
            {
                elements[i] = (float)normalDist.Sample();
                //elements[i] = (float)(rn.NextDouble() * 2 - 1);
            }

            //arrayToTXT(elements);

            Filter = new Tensor(elements, shape);
        }

        private void arrayToTXT(float[] array)
        {
            string path = @"C:\Users\Hendrik\Desktop\WildesDebugging\filter.txt";

            string s = string.Join("\n ", array);

            File.WriteAllText(path, s, Encoding.UTF8);
        }

        #endregion

        #region public methods

        public void Backward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                Tensor newFilters = getTransposedRotatedTensor(Filter);
                int leftPadding = (newFilters.Shape.Axis[0] - 1);
                int topPadding = (newFilters.Shape.Axis[1] - 1);

                int newHeight = inTensor.Shape.Axis[0] + leftPadding * 2;
                int newWidth = inTensor.Shape.Axis[1] +  topPadding * 2;

                float[] paddingElements = new float[newHeight * newWidth * inTensor.Shape.Axis[2]];
                float[] deltas = inTensor.Deltas;

                int row = topPadding;
                int col = leftPadding;
                int channel = 0;
                int colCount = inTensor.Shape.Axis[1];
                int rowCount = inTensor.Shape.Axis[0];

                for (int j = 0; j < deltas.Length;j++)
                {
                    paddingElements[row * newWidth + col + channel * newHeight * newWidth] = deltas[j];

                    col++;
                    if ((col - leftPadding) % colCount == 0)
                    {
                        col = leftPadding;
                        row++;
                        if ((row - topPadding) % rowCount == 0)
                        {
                            channel++;
                            col = leftPadding;
                            row = topPadding;
                        }
                    }
                }

                Tensor oldFilter = Filter;
                Tensor oldBias = Bias;

                Bias = new Tensor(new float[Bias.Elements.Length], Bias.Shape);

                Filter = newFilters;

                Tensor DeltaTensor = new Tensor(paddingElements, new TensorShape(new int[] { newHeight, newWidth, inTensor.Shape.Axis[2] }));
                Tensor tempTensor = new Tensor(new float[outTensor.Elements.Length], outTensor.Shape);
                Forward(new List<Tensor>() { DeltaTensor }, new List<Tensor>() { tempTensor });
                outTensor.Deltas = tempTensor.Elements;

                Filter = new Tensor(oldFilter.Elements, oldFilter.Shape);

                Bias = oldBias;
            }
        }

        public void calculateDeltaWeights(List<Tensor> inTensors, List<Tensor> outTensors, float learningRate)
        {
            for (int i = 0; i < inTensors.Count; i++)
            {
                Tensor oldFilter = Filter;
                Tensor oldBias = Bias;

                Bias = new Tensor(new float[Bias.Elements.Length], Bias.Shape);

                float[] updateArray = new float[0];

                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];

                int filterSize = inTensor.Shape.Axis[0] * inTensor.Shape.Axis[1];
                int matrixSize = outTensor.Shape.Axis[0] * outTensor.Shape.Axis[1];

                for (int j = 0; j < inTensor.Shape.Axis[2]; j++)
                {
                    Filter = new Tensor(inTensor.Deltas.SubArray(j * filterSize, filterSize),
                        new TensorShape(new int[] { inTensor.Shape.Axis[0], inTensor.Shape.Axis[1], 1, 1}));
                    for (int k = 0;k < outTensor.Shape.Axis[2];k++)
                    {
                        Tensor tempTensor = new Tensor(new float[oldFilter.Shape.Axis[0] * oldFilter.Shape.Axis[1]],
                          new TensorShape(new int[] { oldFilter.Shape.Axis[0], oldFilter.Shape.Axis[1], 1, 1 }));

                        Tensor forwardInput = new Tensor(outTensor.Elements.SubArray(k * matrixSize, matrixSize),
                            new TensorShape(new int[] { outTensor.Shape.Axis[0], outTensor.Shape.Axis[1], 1, 1}));

                        Forward(new List<Tensor>() { forwardInput }, new List<Tensor>() { tempTensor });

                        updateArray = updateArray.Concatenate(tempTensor.Elements);
                    }
                }

                for (int j = 0; j < oldFilter.Elements.Length;j++)
                {
                    oldFilter.Elements[j] = oldFilter.Elements[j] - learningRate * updateArray[j];
                }

                Filter = oldFilter;
                Bias = oldBias;

                int length = inTensor.Shape.Axis[0] * inTensor.Shape.Axis[1];
                for (int j =0;j < Filter.Shape.Axis[3];j++)
                {
                    float[] subArray = inTensor.Deltas.SubArray(j * length, length);
                    float sum = subArray.Sum();
                    Bias.Elements[j] = Bias.Elements[j] - learningRate * subArray.Sum();
                }
            }
        }

        public void Forward(List<Tensor> inTensors, List<Tensor> outTensors)
        {
            //iterate batch
            for (int i = 0;i < inTensors.Count; i++)
            {
                Tensor inTensor = inTensors[i];
                Tensor outTensor = outTensors[i];
                int channelNumber = inTensor.Shape.Axis[2];
                int filterNumber = Filter.Shape.Axis[3];

                List<Matrix<float>> inMatrixList = getTransposedMatrix(inTensor);
                List<List<Matrix<float>>> filterMatrixList = getTransposedFilterMatrix(Filter);

                int width = inTensor.Shape.Axis[1];
                int height = inTensor.Shape.Axis[0];
                int filterwidth = Filter.Shape.Axis[1];
                int filterheight = Filter.Shape.Axis[0];

                int index = 0;

                //iterate rows
                for (int j = 0; j <= height - filterheight; j++)
                {
                    //iterate column
                    for (int k = 0; k <= width - filterwidth; k++)
                    {
                        //iterate filters
                        for (int m = 0; m < filterNumber; m++)
                        {
                            //first channel
                            Matrix<float> subMatrix = inMatrixList[0].SubMatrix(k, filterwidth, j, filterheight);
                            Matrix<float> filterMatrix = filterMatrixList[m][0];

                            Matrix<float> newMatrix = filterMatrix.PointwiseMultiply(subMatrix);

                            //iterate second to n-th channel
                            for (int n = 1; n < channelNumber; n++)
                            {
                                subMatrix = inMatrixList[n].SubMatrix(k, filterwidth, j, filterheight);
                                filterMatrix = filterMatrixList[m][n];
                                newMatrix += filterMatrix.PointwiseMultiply(subMatrix);
                            }

                            Vector<float> rowSums = newMatrix.RowSums();
                            float sum = rowSums.Sum() + Bias.Elements[m];

                            int elementNumber = index + m * outTensor.Shape.Axis[0] * outTensor.Shape.Axis[1];

                            outTensor.Elements[elementNumber] = sum;
                        }
                        index++;
                    }
                }
            }
        }

        #endregion

        #region private methods

        private List<Matrix<float>> getTransposedMatrix(Tensor tensor)
        {
            List<Matrix<float>> res = new List<Matrix<float>>();
            int matrixSize = tensor.Shape.Axis[0] * tensor.Shape.Axis[1];
            int elementIndex = 0;

            for (int i = 0; i < tensor.Shape.Axis[2]; i++)
            {
                res.Add(new DenseMatrix(tensor.Shape.Axis[1], tensor.Shape.Axis[0], tensor.Elements.SubArray(elementIndex, matrixSize)));
                elementIndex += matrixSize;
            }
            return res;
        }

        private List<List<Matrix<float>>> getTransposedFilterMatrix(Tensor tensor)
        {
            List<List<Matrix<float>>> res = new List<List<Matrix<float>>>();
            int matrixSize = tensor.Shape.Axis[0] * tensor.Shape.Axis[1];
            int elementIndex = 0;

            for (int i = 0; i < tensor.Shape.Axis[3]; i++)
            {
                List<Matrix<float>> list = new List<Matrix<float>>();
                for (int j = 0; j < tensor.Shape.Axis[2]; j++)
                {
                    list.Add(new DenseMatrix(tensor.Shape.Axis[0], tensor.Shape.Axis[1], tensor.Elements.SubArray(elementIndex, matrixSize)));
                    elementIndex += matrixSize;
                }
                res.Add(list);
            }
            return res;
        }

        private Tensor getTransposedRotatedTensor(Tensor tensor)
        {
            int elementNumber = tensor.Elements.Length;
            float[] newElements = new float[elementNumber];
            int matrixHeight = tensor.Shape.Axis[0];
            int matrixWidth = tensor.Shape.Axis[1];
            int matrixSize = matrixHeight * matrixWidth;

            int m = 0;
            for (int i = 0; i < tensor.Shape.Axis[3]; i++)
            {
                for (int j = 0; j < tensor.Shape.Axis[2]; j++)
                {
                    int offset = i + j * tensor.Shape.Axis[3];

                    for (int k = 0; k < matrixSize; k++)
                    {
                        int elementIndex = k + (i + j + m) * matrixSize;
                        float element = tensor.Elements[elementIndex];
                        int newIndex = (offset + 1) * matrixSize - k - 1;
                        newElements[newIndex] = element;
                    }
                }
                m += tensor.Shape.Axis[2] - 1;
            }


            TensorShape newShape = new TensorShape(new int[] { tensor.Shape.Axis[1], tensor.Shape.Axis[0], tensor.Shape.Axis[3], tensor.Shape.Axis[2] });
            Tensor res = new Tensor(newElements, newShape);

            return res;
        }

        #endregion
    }
}
