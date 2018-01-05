using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cirmex
{
    class GradStep
    {
        public Matrix<double> weightStep;
        public Vector<double> biasStep;

        /// <summary>
        /// Makes GradStep for a layer with w1 outputs, w2 inputs, and b biases
        /// </summary>
        /// <param name="w1"></param>
        /// <param name="w2"></param>
        /// <param name="b"></param>
        public GradStep(int w1, int w2, int b)
        {
            weightStep = Matrix<double>.Build.Dense(w1, w2);
            biasStep = Vector<double>.Build.Dense(b);
        }

        public GradStep(Matrix<double> w, Vector<double> b)
        {
            weightStep = w;
            biasStep = b;
        }
    }
}