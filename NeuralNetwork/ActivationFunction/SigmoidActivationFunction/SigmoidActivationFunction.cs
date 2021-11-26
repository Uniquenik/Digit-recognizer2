using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neural_networks_kubsu.NeuralNetwork.ActivationFunction.SigmoidActivationFunction
{
    class SigmoidActivationFunction: IActivationFunction
    {
        public double Activate(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public double Derivative(double value)
        {
            return Activate(value) * (1 - Activate(value));
        }
    }
}
