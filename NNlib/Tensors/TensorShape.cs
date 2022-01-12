using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NNlib.Tensors
{
    public class TensorShape
    {
        //(rowNumber, columnNumber)

        #region constructor

        public TensorShape(int[] axis)
        {
            Axis = axis;
        }

        #endregion

        #region properties

        public int[] Axis { get; set; }

        #endregion

        public override bool Equals(Object obj)
        {
            //Check for null and compare run-time types.
            if ((obj == null) || !this.GetType().Equals(obj.GetType()))
            {
                return false;
            }
            else
            {
                TensorShape t = (TensorShape)obj;
                if (Axis.Length != t.Axis.Length)
                {
                    return false;
                }
                bool res = true;
                for (int i = 0; i < Axis.Length;i++)
                {
                    res = res && Axis[i] == t.Axis[i];
                }
                return res;
            }
        }

    }
}
