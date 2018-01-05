using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cirmex
{
    class Layer
    {
        public Vector<double> neurons;

        public Layer()
        {
            //this needs to exist so you can create derived layers with neurons initialized by a starting layer and not just passed in
        }

        public Layer(double[] n)
        {
            neurons = Vector<double>.Build.Dense(n);
        }

        public int size()
        {
            return neurons.Count;
        }

        public double getActivationFormula(int n)
        {
            return neurons[n]; //just gets activation of neuron, for so backpropogation formula stops here
        }
    }
}
