using System;
using MathNet.Numerics.LinearAlgebra;
using System.Windows;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cirmex
{
    class DerivedLayer : Layer
    {
        public Matrix<double> weights;
        public Vector<double> biases;
        Layer input;
        

        public DerivedLayer(Layer init, int size)
        {
            neurons = neurons = Vector<double>.Build.Dense(size);
            input = init;
            weights = Matrix<double>.Build.Dense(size, init.size()); //# of columns: number of inputs, # of rows: number of outputs.
                                                                     //so creates a net that takes all inputs from its previous layer and returns a number of outputs equal to its size
            biases = Vector<double>.Build.Dense(size);
        }
        
        /// <summary>
        /// Calculates this layer's neuron values from its linked layers' values.
        /// </summary>
        public void deriveValues()
        {
            neurons = (weights.Multiply(input.neurons) + biases).Map(Compressions.ReLU); //for each output neuron, multiply every input neuron by a weight and sum, then add a bias and apply ReLU
        }

        /// <summary>
        /// Randomly scrambles all weights and biases, to reset at the first round of testing.
        /// </summary>
        public void randomize()
        {
            weights = Matrix<double>.Build.Random(weights.RowCount, weights.ColumnCount);
            biases = Vector<double>.Build.Random(biases.Count);
        }

        public double numFactors()
        {
            return (weights.RowCount * weights.ColumnCount) + biases.Count;
        }
    }
}