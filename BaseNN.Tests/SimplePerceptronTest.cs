using System;
using Xunit;
using BaseNN;
using System.Numerics;

namespace BaseNN.Tests
{
    public class SimplePerceptronTest
    {
        // bias and weights for simple tests
        static float Bias = -2;
        static float[] weights = {1,2,0,0,0,0,0,0};
        static Vector<float> Weights = new Vector<float>(weights);

        // perceptron for simple tests
        Neuron p = new Neuron(Bias, Weights);

        [Theory]
        [InlineData(1,2)]
        [InlineData(2,1)]
        [InlineData(1,1)]
        public void SimpleTestShouldYield1(float v0, float v1)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            inputs[1] = v1;
            Assert.Equal(1, p.Feed(new Vector<float>(inputs)));
        }

        [Theory]
        [InlineData(1,0)]
        [InlineData(0,1)]
        [InlineData(0,0)]
        public void SimpleTestShouldYield0(float v0, float v1)
        {
            float[] inputs = new float[8];
            inputs[0] = v0;
            inputs[1] = v1;
            Assert.Equal(0, p.Feed(new Vector<float>(inputs)));
        }
    }
}
