using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace NNlib.Extensions
{
    public static class ArrayFlattener
    {
		public static T[] Flatten<T>(this Array data)
		{
			var list = new List<T>();
			var stack = new Stack<IEnumerator>();
			stack.Push(data.GetEnumerator());
			do
			{
				for (var iterator = stack.Pop(); iterator.MoveNext();)
				{
					if (iterator.Current is Array)
					{
						stack.Push(iterator);
						iterator = (iterator.Current as IEnumerable).GetEnumerator();
					}
					else
						list.Add((T)iterator.Current);
				}
			}
			while (stack.Count > 0);
			return list.ToArray();
		}
	}
}
