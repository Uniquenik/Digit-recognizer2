using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using System.Xml;
using neural_networks_kubsu.NeuralNetwork.ActivationFunction.TanhActivationFunction;
using neural_networks_kubsu.NeuralNetwork.LossFunction.EuclideanDistanceLoss;
using neural_networks_kubsu.NeuralNetwork.WeightsInitializer.DefaultWeightsInitializer;
using static neural_networks_kubsu.NeuralNetwork.NeuralNetwork;

namespace neural_networks_kubsu
{
    public partial class FormMain : Form
    {
        private readonly double[] _inputArray = new double[15];

        public static Label LabelNeurons;
        public static TextBox TextTrain;
        public static NumericUpDown Count;
        private NeuralNetwork.NeuralNetwork _nn;

        private double[][] _inputData =
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

        private double[][] _outputData =
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
            TextTrain = textBox1;
            Count = numericUpDown1;
            CreateNeuralNetwork(NeuralNetwork.NeuralNetwork.InitType.SETLOCAL);
        }

        private void button15_Click(object sender, EventArgs e)
        {
            var thread2 = new Thread(Fit);
            thread2.Start();
        }

        private void Fit()
        {
            _nn.Fit(_inputData, _outputData, Convert.ToInt32(Count.Value) , 0.01);
            Predict();
        }

        private void button16_Click(object sender, EventArgs e)
        {
            CreateNeuralNetwork(NeuralNetwork.NeuralNetwork.InitType.SETLOCAL);
            //_nn.Initialize(NeuralNetwork.NeuralNetwork.InitType.SET, new DefaultWeightsInitializer());
        }

        private void CreateNeuralNetwork(InitType type)
        {
            _nn = Builder()
                .InputLayer(15)
                .HiddenLayer(73, new TanhActivationFunction())
                .HiddenLayer(33, new TanhActivationFunction())
                .OutputLayer(10, new TanhActivationFunction())
                .LossFunction(new EuclideanDistanceLoss())
                .Build();
            _nn.Initialize(type, new DefaultWeightsInitializer());
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

        private void button18_Click(object sender, EventArgs e)
        {
            XmlDocument memory_doc = new XmlDocument();
            if (!File.Exists(System.IO.Path.Combine("Resources", $"data_set.xml")))
            {
                XmlElement element1 = memory_doc.CreateElement("", "sets", "");
                memory_doc.AppendChild(element1);
                memory_doc.Save(System.IO.Path.Combine("Resources", $"data_set.xml"));
                memory_doc.Load(System.IO.Path.Combine("Resources", $"data_set.xml"));
            }
            else
            {
                memory_doc.Load(System.IO.Path.Combine("Resources", $"data_set.xml"));
            }
            XmlNode el1 = memory_doc.FirstChild;
            XmlElement elementtemp = memory_doc.CreateElement(string.Empty, "set", string.Empty);
            XmlElement elementintemp2 = memory_doc.CreateElement(string.Empty, "result", string.Empty);
            XmlElement elementintemp1 = memory_doc.CreateElement(string.Empty, "input", string.Empty);
            string outputData = "";
            for (int i = 0; i <= 9; i++) {
                if (i == Convert.ToInt32(TextTrain.Text))
                    outputData += 1.0.ToString() + " ";
                else outputData += 0.0.ToString() + " ";
            }
            outputData = outputData.Substring(0, outputData.Length - 1);
            XmlText textintemp = memory_doc.CreateTextNode(outputData);
            elementintemp2.AppendChild(textintemp);
            XmlText texttemp = memory_doc.CreateTextNode(string.Join(" ", _inputArray));
            elementintemp1.AppendChild(texttemp);
            elementtemp.AppendChild(elementintemp1);
            elementtemp.AppendChild(elementintemp2);
            //elementtemp.AppendChild(texttemp);
            el1.AppendChild(elementtemp);
            memory_doc.Save(System.IO.Path.Combine("Resources", $"data_set.xml"));
        }

        private void button20_Click(object sender, EventArgs e)
        {
            //CreateNeuralNetwork(NeuralNetwork.NeuralNetwork.InitType.GET);
            _nn.Initialize(NeuralNetwork.NeuralNetwork.InitType.GET, new DefaultWeightsInitializer());
            Predict();
            Evaluate();
        }

        private void button19_Click(object sender, EventArgs e)
        {
            XmlDocument memory_doc = new XmlDocument();
            if (!File.Exists(System.IO.Path.Combine("Resources", $"data_set.xml")))
            {
                XmlElement element1 = memory_doc.CreateElement("", "sets", "");
                memory_doc.AppendChild(element1);
                memory_doc.Save(System.IO.Path.Combine("Resources", $"data_set.xml"));
                memory_doc.Load(System.IO.Path.Combine("Resources", $"data_set.xml"));
            }
            else
            {
                memory_doc.Load(System.IO.Path.Combine("Resources", $"data_set.xml"));
            }
            var count = memory_doc.SelectNodes("sets/set").Count;
            _inputData = new double[count][];
            Array.Clear(_inputData, 0, count);
            _outputData = new double[count][];
            Array.Clear(_outputData, 0, count);
            XmlElement memory_el = memory_doc.DocumentElement;
            //MessageBox.Show(memory_el.ChildNodes.Item(0).LastChild.InnerText.ToString());
            for (int i = 0; i < count; i++) {
                _outputData[i] = memory_el.ChildNodes.Item(i).LastChild.InnerText.Split(' ').Select(double.Parse).ToArray();
                _inputData[i] = memory_el.ChildNodes.Item(i).FirstChild.InnerText.Split(' ').Select(double.Parse).ToArray();
            }

           /* XmlElement memory_el = memory_doc.DocumentElement;
            for (int l = 0; l < weights.GetLength(0); ++l)
                for (int k = 0; k < weights[0].GetLength(0); ++k)
                {

                    weights[l][k] = double.Parse(memory_el.ChildNodes.Item(k + weights[0].Length * l).InnerText.Replace(',', '.'), System.Globalization.CultureInfo.InvariantCulture);
                }*/
            //_inputData.Clear();
        }
    }
}