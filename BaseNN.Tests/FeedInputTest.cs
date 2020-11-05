using System;
using Xunit;
using BaseNN;
using System.Numerics;

namespace BaseNN.Tests
{
    public class FeedInputPositionTest
    {
        // bias and weights for logical `AND` tests using
        // the last two fields of the weights and input vector
        static float Bias = -1.5F;
        static float[] weights = {0,0,0,0,0,0,1,1};
        static Vector<float> Weights = new Vector<float>(weights);
        Neuron p = new Neuron(Bias,  Weights);

        [Theory]
        [InlineData(1,1)]
        public void TestLogicalAndYieldsTrue(float v6, float v7)
        {
            float[] inputs = new float[8];
            inputs[6] = v6;
            inputs[7] = v7;
            Assert.Equal(1, p.Feed(new Vector<float>(inputs)));
        }

        [Theory]
        [InlineData(1,0)]
        [InlineData(0,1)]
        [InlineData(0,0)]
        public void TestLogicalAndYieldsFalse(float v6, float v7)
        {
            float[] inputs = new float[8];
            inputs[6] = v6;
            inputs[7] = v7;
            Assert.Equal(0, p.Feed(new Vector<float>(inputs)));
        }
    }
}