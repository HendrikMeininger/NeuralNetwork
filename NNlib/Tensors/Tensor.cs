using System;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Tensors
{
    public class Tensor
    {
        #region constructor

        public Tensor(float[] elements, TensorShape shape)
        {
            Elements = elements;
            Shape = shape;
        }

        #endregion

        #region properties

        public float[] Elements { get; set; }

        public TensorShape Shape { get; set; }

        public float[] Deltas
        {
            get
            {
                if (_deltas == null)
                {
                    initDeltas();
                }
                return _deltas;
            }
            set => _deltas = value;
        }

        private float[] _deltas;

        #endregion

        #region private methods

        private void initDeltas()
        {
            Deltas = new float[Elements.Length];
        }

        #endregion
    }
}
