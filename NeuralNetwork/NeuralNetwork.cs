using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml;
using neural_networks_kubsu.NeuralNetwork.ActivationFunction;
using neural_networks_kubsu.NeuralNetwork.Layer.HiddenLayer;
using neural_networks_kubsu.NeuralNetwork.Layer.InputLayer;
using neural_networks_kubsu.NeuralNetwork.Layer.OutputLayer;
using neural_networks_kubsu.NeuralNetwork.LossFunction;
using neural_networks_kubsu.NeuralNetwork.Neuron;
using neural_networks_kubsu.NeuralNetwork.WeightsInitializer;

namespace neural_networks_kubsu.NeuralNetwork
{
    public class NeuralNetwork
    {
        private IInputLayer _inputLayer;
        private IOutputLayer OutputLayer => (IOutputLayer) _layers.Last();
        private readonly List<IHiddenLayer> _layers = new();
        private ILossFunction _lossFunction;

        public double[] Predict(double[] inputData)
        {
            _inputLayer.Feed(inputData);
            foreach (var layer in _layers)
            {
                layer.ComputeNeurons();
            }

            return OutputLayer.Result;
        }

        public enum InitType
        {
            GET, SET, WORKSET, SETLOCAL
        }

        public void Files(InitType mode, int index, double [][] weights, INeuron [] neurons, double lastneurons) {
            XmlDocument memory_doc = new XmlDocument();
            string name = $"{index}_layer_memory.xml";
            if (!File.Exists(System.IO.Path.Combine("Resources", name)))
            {
                XmlElement element1 = memory_doc.CreateElement("", "Weights", "");
                memory_doc.AppendChild(element1);
                memory_doc.Save(System.IO.Path.Combine("Resources", name));
                memory_doc.Load(System.IO.Path.Combine("Resources", name));
            }
            else
            {
                memory_doc.Load(System.IO.Path.Combine("Resources", name));
            }
            if (mode == InitType.GET)
            {
                XmlElement memory_el = memory_doc.DocumentElement;
                for (int l = 0; l < weights.GetLength(0); ++l)
                    for (int k = 0; k < weights[0].GetLength(0); ++k)
                    {

                        weights[l][k] = double.Parse(memory_el.ChildNodes.Item(k + weights[0].Length * l).InnerText.Replace(',', '.'), System.Globalization.CultureInfo.InvariantCulture);
                    }
            }
            else
            {
                XmlNode el1 = memory_doc.FirstChild;
                memory_doc.FirstChild.RemoveAll();
                for (int l = 0; l < neurons.Length; ++l)
                {
                    for (int k = 0; k < lastneurons + 1; ++k)
                    {
                        XmlElement elementtemp = memory_doc.CreateElement(string.Empty, "weight", string.Empty);
                        XmlText texttemp = memory_doc.CreateTextNode(weights[l][k].ToString());
                        elementtemp.AppendChild(texttemp);
                        el1.AppendChild(elementtemp);
                    }
                }
            }
            memory_doc.Save(System.IO.Path.Combine("Resources", name));
        }

        public void Initialize(InitType mode, IWeightsInitializer weightsInitializer)
        {
            _layers[0].PreviousLayer = _inputLayer;
            if (mode == InitType.SET || mode == InitType.SETLOCAL)
            {
                _layers[0].Weights = weightsInitializer.Initialize(
                    _inputLayer.Neurons.Length,
                    _layers[0].Neurons.Length,
                    1
                );
            }
            else if (mode == InitType.GET)
            {
                var weights = new double[_layers[0].Neurons.Length][];
                foreach (var i in Enumerable.Range(0, _layers[0].Neurons.Length))
                {
                    weights[i] = new double[15 + 1];
                }
                _layers[0].Weights = weights;
            }
            if (mode != InitType.SETLOCAL) Files(mode, 0, _layers[0].Weights, _layers[0].Neurons, 15);
            
            for (var layerIndex = 1; layerIndex < _layers.Count; layerIndex++)
            {
                if (mode == InitType.SET || mode == InitType.SETLOCAL)
                {
                    _layers[layerIndex].Weights = weightsInitializer.Initialize(
                        _layers[layerIndex - 1].Neurons.Length,
                        _layers[layerIndex].Neurons.Length,
                        layerIndex + 1
                    );
                }

                else if (mode == InitType.GET)
                {
                    var weights = new double[_layers[layerIndex].Neurons.Length][];
                    foreach (var i in Enumerable.Range(0, _layers[layerIndex].Neurons.Length))
                    {
                        weights[i] = new double[_layers[layerIndex - 1].Neurons.Length + 1];
                    }
                    _layers[layerIndex].Weights = weights;
                }
                if (mode != InitType.SETLOCAL) Files(mode, layerIndex, _layers[layerIndex].Weights, _layers[layerIndex].Neurons, _layers[layerIndex - 1].Neurons.Length);
            }

            for (var layerIndex = 0; layerIndex < _layers.Count - 1; layerIndex++)
            {
                _layers[layerIndex].NextLayer = _layers[layerIndex + 1];
            }

            for (var layerIndex = 1; layerIndex <= _layers.Count - 1; layerIndex++)
            {
                _layers[layerIndex].PreviousLayer = _layers[layerIndex - 1];
            }

            foreach (var layer in _layers)
            {
                layer.Initialize();
            }
        }

        private void CorrectWeights(double learningRate)
        {
            foreach (var layer in _layers)
            {
                layer.CorrectWeights(learningRate);
            }
        }

        private void ComputeDelta(double[] data)
        {
            OutputLayer.ComputeDelta(data);
            for (var i = _layers.Count - 2; i >= 0; i--)
            {
                _layers[i].ComputeDelta();
            }
        }


        public void Fit(double[][] inputBatch, double[][] outputBatch, int epochs, double learningRate)
        {
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                for (var i = 0; i < inputBatch.Length; i++)
                {
                    Predict(inputBatch[i]);
                    ComputeDelta(outputBatch[i]);
                    CorrectWeights(learningRate);
                }
                if (epoch % 10 == 0)
                {
                    FormMain.LabelNeurons.Invoke((Action)delegate { FormMain.LabelEp.Text = "Ep: " + epoch.ToString(); });
                    FormMain.LabelNeurons.Invoke((Action)delegate { FormMain.LabelNeurons.Text = "Loss: " + Evaluate(inputBatch, outputBatch).ToString(); });  
                }
            }
        }

        public double Evaluate(double[][] inputBatch, double[][] outputBatch)
        {
            var s = 0.0;
            for (var i = 0; i < inputBatch.Length; i++)
            {
                s += Math.Pow(_lossFunction.ComputeLoss(Predict(inputBatch[i]), outputBatch[i]), 2.0);
            }
            return Math.Sqrt(s);
        }

        public static NeuralNetworkBuilder Builder()
        {
            return new NeuralNetworkBuilder();
        }

        public class NeuralNetworkBuilder
        {
            private readonly NeuralNetwork _instance;

            internal NeuralNetworkBuilder()
            {
                _instance = new NeuralNetwork();
            }

            public NeuralNetworkBuilder InputLayer(int units)
            {
                _instance._inputLayer = new InputLayer(units);
                return this;
            }

            public NeuralNetworkBuilder HiddenLayer(int units, IActivationFunction activationFunction)
            {
                _instance._layers.Add(new HiddenLayer(units) {ActivationFunction = activationFunction});
                return this;
            }

            public NeuralNetworkBuilder OutputLayer(int units, IActivationFunction activationFunction)
            {
                _instance._layers.Add(new OutputLayer(units) {ActivationFunction = activationFunction});
                return this;
            }

            public NeuralNetworkBuilder LossFunction(ILossFunction lossFunction)
            {
                _instance._lossFunction = lossFunction;
                return this;
            }

            public NeuralNetwork Build()
            {
                return _instance;
            }
        }
    }
}