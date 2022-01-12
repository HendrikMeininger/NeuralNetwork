using NNlib.Extensions;
using NNlib.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace NNlib.Layers
{
    public class InputLayer<T>
    {
        #region constructor

        public InputLayer()
        {
        }

        #endregion

        #region public methods

        public List<Tensor> Forward<T>(List<T> rawData)
        {
            List<Tensor> res = new List<Tensor>();

            foreach (T data in rawData)
            {
                switch (data)
                {
                    case int[] i1:
                        res.Add(tensorsFromIntArray(i1));
                        break;
                    case int[,] i2:
                        res.Add(tensorsFrom2DIntArray(i2));
                        break;
                    case int[,,] i3:
                        res.Add(tensorsFrom3DIntArray(i3));
                        break;
                    case float[] f1:
                        res.Add(tensorsFromFloatArray(f1));
                        break;
                    case float[,] f2:
                        res.Add(tensorsFrom2DFloatArray(f2));
                        break;
                    case float[,,] f3:
                        res.Add(tensorsFrom3DFloatArray(f3));
                        break;
                    case Image i:
                        res.Add(tensorsFromImage(i));
                        break;
                    case byte[][] b:
                        res.Add(tensorFromByteArray(b));
                        break;
                    case byte[,] b:
                        res.Add(tensorFromByteArray(b));
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }

            return res;
        }

        #endregion

        #region TensorCreators

        private Tensor tensorsFromIntArray(int[] arr)
        {
            TensorShape shape = new TensorShape(new int[] { 1, arr.GetLength(0) });
            float[] elements = intArrToFloatArr(arr);

            return new Tensor(elements, shape);
        }

        private Tensor tensorsFrom2DIntArray(int[,] arr)
        {
            TensorShape shape = new TensorShape(new int[] { arr.GetLength(0), arr.GetLength(1) });
            float[,] floatArray = intArrToFloatArr(arr);
            float[] elements = ArrayFlattener.Flatten<float>(floatArray);

            return new Tensor(elements, shape);
        }

        private Tensor tensorsFrom3DIntArray(int[,,] arr)
        {
            TensorShape shape = new TensorShape(new int[] { arr.GetLength(0), arr.GetLength(1), arr.GetLength(2) });
            float[,,] floatArray = intArrToFloatArr(arr);
            float[] elements = ArrayFlattener.Flatten<float>(floatArray);

            return new Tensor(elements, shape);
        }

        private Tensor tensorsFromFloatArray(float[] arr)
        {
            TensorShape shape = new TensorShape(new int[] { 1, arr.GetLength(0) });

            return new Tensor(arr, shape);
        }

        private Tensor tensorsFrom2DFloatArray(float[,] arr)
        {
            TensorShape shape = new TensorShape(new int[] { arr.GetLength(0), arr.GetLength(1) });
            float[] elements = ArrayFlattener.Flatten<float>(arr);

            return new Tensor(elements, shape);
        }

        private Tensor tensorsFrom3DFloatArray(float[,,] arr)
        {
            TensorShape shape = new TensorShape(new int[] { arr.GetLength(0), arr.GetLength(1), arr.GetLength(2) });
            float[] elements = ArrayFlattener.Flatten<float>(arr);

            return new Tensor(elements, shape);
        }

        private Tensor tensorsFromImage(Image image)
        {
            Bitmap bmp = new Bitmap(image);
            int height = bmp.Height;
            int width = bmp.Width;

            TensorShape shape = new TensorShape(new int[] { height, width, 3 });

            float[] elements = new float[height * width];

            for (int i = 0;i < height;i++)
            {
                for (int j = 0;j < width;j++)
                {
                    Color clr = bmp.GetPixel(i, j); 
                    int red = clr.R;
                    int green = clr.G;
                    int blue = clr.B;

                    elements[i * width + j] = (red + green + blue) / 3;
                }
            }

            return new Tensor(elements, shape);
        }

        private Tensor tensorFromByteArray(byte[][] arr)
        {
            TensorShape shape = new TensorShape(new int[] { arr.Length, arr[0].Length });
            float[,] floatArray = byteArrayToFloatArray(arr);
            float[] elements = ArrayFlattener.Flatten<float>(floatArray);

            return new Tensor(elements, shape);
        }

        private Tensor tensorFromByteArray(byte[,] arr)
        {
            TensorShape shape = new TensorShape(new int[] { 1, arr.GetLength(0) * arr.GetLength(1) });
            float[,] floatArray = byteArrayToFloatArray(arr);
            float[] flatArray = ArrayFlattener.Flatten<float>(floatArray);

            return new Tensor(flatArray, shape);
        }

        #endregion

        #region private methods

        private float[] intArrToFloatArr(int[] arr)
        {
            float[] res = new float[arr.Length];

            for (int i = 0; i < arr.Length; ++i)
            {
                res[i] = arr[i];
            }

            return res;
        }

        private float[,] intArrToFloatArr(int[,] arr)
        {
            float[,] res = new float[arr.GetLength(0), arr.GetLength(1)];

            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    res[i, j] = arr[i, j];
                }
            }

            return res;
        }

        private float[,,] intArrToFloatArr(int[,,] arr)
        {
            float[,,] res = new float[arr.GetLength(0), arr.GetLength(1), arr.GetLength(2)];

            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    for (int k = 0; k < arr.GetLength(1); k++)
                    {
                        res[i, j, k] = arr[i, j, k];
                    }
                }
            }

            return res;
        }

        private float[,] byteArrayToFloatArray(byte[][] arr)
        {
            float[,] res = new float[arr.GetLength(0), arr.GetLength(1)];

            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    res[i, j] = arr[i][j];
                }
            }

            return res;
        }

        private float[,] byteArrayToFloatArray(byte[,] arr)
        {
            float[,] res = new float[arr.GetLength(0), arr.GetLength(1)];

            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    res[i, j] = arr[i, j] / 255f;
                }
            }

            return res;
        }

        #endregion

    }
}
