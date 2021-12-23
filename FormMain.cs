using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using System.Xml;
using neural_networks_kubsu.NeuralNetwork.ActivationFunction.SigmoidActivationFunction;
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
        public static Label LabelCount;
        public static Label LabelAnswer;
        public static Label LabelEp;
        public static Label LabelFileName;
        public static Label LabelFileTrain;
        public static TextBox TextTrain;
        public static NumericUpDown Count;
        public static Chart Chart1;
        public static Series ser;
        private NeuralNetwork.NeuralNetwork _nn;
        private string addTrainFile = null;

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
            LabelCount = label1;
            LabelEp = label2;
            LabelAnswer = label3;
            LabelFileName = label4;
            LabelFileTrain = label5;
            Chart1 = chart1;
            CreateNeuralNetwork(NeuralNetwork.NeuralNetwork.InitType.SETLOCAL);
        }
        private void CreateNeuralNetwork(InitType type)
        {
            _nn = Builder()
                .InputLayer(15)
                .HiddenLayer(73, new TanhActivationFunction())
                .HiddenLayer(33, new TanhActivationFunction())
                .OutputLayer(10, new SigmoidActivationFunction())
                .LossFunction(new EuclideanDistanceLoss())
                .Build();
            _nn.Initialize(type, new DefaultWeightsInitializer());
            Predict();
            Evaluate();
        }

        private void button15_Click(object sender, EventArgs e)
        {
            var thread2 = new Thread(Fit);
            thread2.Start();
        }

        private void Fit()
        {
            string[] seriesArray = { "chart" };
            if (ser == null)
            {
                if (InvokeRequired)
                {
                    Invoke(new Action(() =>
                    {
                        ser = Chart1.Series.Add(seriesArray[0]);
                        Chart1.Series["chart"].LegendText = "sas";
                        Chart1.Series["chart"].ChartType = SeriesChartType.Spline;
                        Chart1.Series["chart"].BorderWidth = 3;
                        Chart1.ChartAreas[0].AxisX.MajorGrid.LineDashStyle = ChartDashStyle.DashDot;
                        Chart1.ChartAreas[0].AxisY.MajorGrid.LineDashStyle = ChartDashStyle.DashDot;
                        Chart1.Palette = ChartColorPalette.Excel;
                        Chart1.Series[0].IsVisibleInLegend = false;
                        Chart1.Series["chart"].IsVisibleInLegend = false;
                    }));
                }
                else
                {
                    ser = Chart1.Series.Add(seriesArray[0]);
                    Chart1.Series["chart"].LegendText = "sas";
                    Chart1.Series["chart"].ChartType = SeriesChartType.Spline;
                    Chart1.Series["chart"].BorderWidth = 3;
                    Chart1.ChartAreas[0].AxisX.MajorGrid.LineDashStyle = ChartDashStyle.DashDot;
                    Chart1.ChartAreas[0].AxisY.MajorGrid.LineDashStyle = ChartDashStyle.DashDot;
                    Chart1.Palette = ChartColorPalette.Excel;
                    Chart1.Series[0].IsVisibleInLegend = false;
                    Chart1.Series["chart"].IsVisibleInLegend = false;
                }
            }
            _nn.Fit(_inputData, _outputData, Convert.ToInt32(Count.Value), 0.05);
            Predict();
        }

        private void button16_Click(object sender, EventArgs e)
        {
            CreateNeuralNetwork(InitType.SETLOCAL);
        }

        private void Evaluate()
        {
            labelEvaluationValue.Text = "Loss: " + _nn.Evaluate(_inputData, _outputData);
        }

        private void Predict()
        {
            var prediction = _nn.Predict(_inputArray);
            var s = "";
            for (var i = 0; i < 10; i++)
            {
                s += i + ": " + prediction[i] + "\n";
            }
            string ans = "N";
            if (prediction.Max() > 0.2) ans = Array.IndexOf(prediction, prediction.Max()).ToString();

            if (InvokeRequired)
            {
                Invoke(new Action(() =>
                {
                    labelStatus.Text = s;
                    LabelAnswer.Text = ans;
                }));
            }
            else
            {
                labelStatus.Text = s;
                LabelAnswer.Text = ans;
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
            if (addTrainFile == null) {
                OpenFileDialog openFileDialog1 = new OpenFileDialog();
                if (openFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    addTrainFile = openFileDialog1.FileName;
                }
            }

            XmlDocument memory_doc = new XmlDocument();
            if (!File.Exists(System.IO.Path.Combine("Resources",addTrainFile)))
            {
                XmlElement element1 = memory_doc.CreateElement("", "sets", "");
                memory_doc.AppendChild(element1);
                memory_doc.Save(System.IO.Path.Combine("Resources", addTrainFile));
                memory_doc.Load(System.IO.Path.Combine("Resources", addTrainFile));
            }
            else
            {
                memory_doc.Load(System.IO.Path.Combine("Resources", addTrainFile));
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
            el1.AppendChild(elementtemp);
            memory_doc.Save(System.IO.Path.Combine("Resources", addTrainFile));
            var count = memory_doc.SelectNodes("sets/set").Count;
            label1.Text = count.ToString()+" train images";
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
            OpenFileDialog openFileDialog1 = new OpenFileDialog();
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                setTrainData(openFileDialog1.FileName);
                LabelFileTrain.Text = "Train from: "+openFileDialog1.FileName;
            }
        }

        private void setTrainData(string name) {
            XmlDocument memory_doc = new XmlDocument();
            if (!File.Exists(System.IO.Path.Combine("Resources", name)))
            {
                XmlElement element1 = memory_doc.CreateElement("", "sets", "");
                memory_doc.AppendChild(element1);
                memory_doc.Save(System.IO.Path.Combine("Resources", name));
                memory_doc.Load(System.IO.Path.Combine("Resources", name));
            }
            else
            {
                memory_doc.Load(System.IO.Path.Combine("Resources", name));
            }
            var count = memory_doc.SelectNodes("sets/set").Count;
            _inputData = new double[count][];
            Array.Clear(_inputData, 0, count);
            _outputData = new double[count][];
            Array.Clear(_outputData, 0, count);
            XmlElement memory_el = memory_doc.DocumentElement;
            for (int i = 0; i < count; i++)
            {
                _outputData[i] = memory_el.ChildNodes.Item(i).LastChild.InnerText.Split(' ').Select(double.Parse).ToArray();
                _inputData[i] = memory_el.ChildNodes.Item(i).FirstChild.InnerText.Split(' ').Select(double.Parse).ToArray();
            }
            LabelCount.Text = count.ToString()+" train images";

        }

        private void button21_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog1 = new OpenFileDialog();
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                addTrainFile = openFileDialog1.FileName;
                LabelFileName.Text = "Add data in: " + addTrainFile;
            }
        }
        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {

        }

        private void labelEvaluationValue_Click(object sender, EventArgs e)
        {

        }

        private void button22_Click(object sender, EventArgs e)
        {
            SaveFileDialog saveFileDialog1 = new SaveFileDialog();

            saveFileDialog1.Filter = "XML (*.xml)|*.xml";
            saveFileDialog1.RestoreDirectory = true;

            if (saveFileDialog1.ShowDialog() == DialogResult.OK)
            {
                addTrainFile = saveFileDialog1.FileName;
                LabelFileName.Text = "Add data in: "+addTrainFile;
            }
        }
    }
}