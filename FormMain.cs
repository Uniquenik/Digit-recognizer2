﻿using System;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using neural_networks_kubsu.NeuralNetwork.ActivationFunction.TanhActivationFunction;
using neural_networks_kubsu.NeuralNetwork.LossFunction.EuclideanDistanceLoss;
using neural_networks_kubsu.NeuralNetwork.WeightsInitializer.DefaultWeightsInitializer;

namespace neural_networks_kubsu
{
    public partial class FormMain : Form
    {
        private readonly double[] _inputArray = new double[15];

        public static Label LabelNeurons;
        private NeuralNetwork.NeuralNetwork _nn;

        private readonly double[][] _inputData =
        {
            new[]
            {
                1.0, 1.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            new[]
            {
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0
            },
            new[]
            {
                1.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 0.0, 0.0,
                1.0, 1.0, 1.0
            },
            new[]
            {
                1.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            new[]
            {
                1.0, 0.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0
            },
            new[]
            {
                1.0, 1.0, 1.0,
                1.0, 0.0, 0.0,
                1.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            new[]
            {
                1.0, 1.0, 1.0,
                1.0, 0.0, 0.0,
                1.0, 1.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            new[]
            {
                1.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0
            },
            new[]
            {
                1.0, 1.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            new[]
            {
                1.0, 1.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            new[]
            {
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            }
        };

        private readonly double[][] _outputData =
        {
            new[] {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            new[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
            new[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            new[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        };


        public FormMain()
        {
            InitializeComponent();
            LabelNeurons = labelEvaluationValue;
            CreateNeuralNetwork();
        }

        private void button15_Click(object sender, EventArgs e)
        {
            var thread2 = new Thread(Fit);
            thread2.Start();
        }

        private void Fit()
        {
            _nn.Fit(_inputData, _outputData, 500, 0.01);
            Predict();
        }

        private void button16_Click(object sender, EventArgs e)
        {
            CreateNeuralNetwork();
        }

        private void CreateNeuralNetwork()
        {
            _nn = NeuralNetwork.NeuralNetwork.Builder()
                .InputLayer(15)
                .HiddenLayer(73, new TanhActivationFunction())
                .HiddenLayer(33, new TanhActivationFunction())
                .OutputLayer(10, new TanhActivationFunction())
                .LossFunction(new EuclideanDistanceLoss())
                .Build();
            _nn.Initialize(NeuralNetwork.NeuralNetwork.InitType.GET, new DefaultWeightsInitializer());
            Predict();
            Evaluate();
        }

        private void Evaluate()
        {
            labelEvaluationValue.Text = "Loss: " + _nn.Evaluate(_inputData, _outputData);
        }

        private void Predict()
        {
            var prediction = _nn.Predict(_inputArray);
            var s = "Prediction:\n";
            for (var i = 0; i < 10; i++)
            {
                s += i + ": " + prediction[i] + "\n";
            }

            if (InvokeRequired)
            {
                Invoke(new Action(() =>
                {
                    labelStatus.Text = s;
                }));
            }
            else
            {
                labelStatus.Text = s;
            }
            
        }

        private void btn_click(Button btn)
        {
            var buttonId = btn.TabIndex;
            btn.BackColor = _inputArray[buttonId] == 0.0 ? Color.Gray: Color.White;
            _inputArray[buttonId] = Math.Abs(1.0 - _inputArray[buttonId]);
            Predict();
        }
        
        private void button0_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }
        
        private void button6_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button7_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button8_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button9_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button10_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button11_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button12_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button13_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button14_Click(object sender, EventArgs e)
        {
            btn_click(sender as Button);
        }

        private void button17_Click(object sender, EventArgs e)
        {
            _nn.Initialize(NeuralNetwork.NeuralNetwork.InitType.WORKSET, new DefaultWeightsInitializer());
        }
    }
}