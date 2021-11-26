using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neural_networks_kubsu.NeuralNetwork.ActivationFunction
{
    public class LeakyReLUActivationFunction : IActivationFunction
    {
        public double Activate(double value)
        {
            return (value >= 0) ? value : 0.01d * value;
        }

        public double Derivative(double value)
        {
            return (value >= 0) ? 1 : 0.01d;
        }
    }
}
