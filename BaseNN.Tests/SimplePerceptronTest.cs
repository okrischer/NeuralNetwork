using System;
using Xunit;
using BaseNN;

namespace BaseNN.Tests
{
    public class SimplePerceptronTest
    {
        // bias and weights for simple tests
        static float bias = -2;
        static float[] weights = {1,2};

        // perceptron for simple tests
        Neuron perceptron = new Neuron(bias,  weights);

        [Fact]
        public void ShouldYieldException()
        {
            float[] input = {1};
            try
            {
                var result = perceptron.Feed(input);
            }
            catch (OperationCanceledException oce)
            {
                Assert.Equal("length of input doesn`t match length of weights"
                , oce.Message);
            }
        }

        [Theory]
        [InlineData(1,2)]
        [InlineData(2,1)]
        [InlineData(1,1)]
        public void SimpleTestShouldYield1(float value1, float value2)
        {
            float[] input = {value1, value2};
            var result = perceptron.Feed(input);
            Assert.Equal(1, result);
        }
        [Theory]
        [InlineData(1,0)]
        [InlineData(0,1)]
        [InlineData(0,0)]
        public void SimpleTestShouldYield0(float value1, float value2)
        {
            float[] input = {value1, value2};
            var result = perceptron.Feed(input);
            Assert.Equal(0, result);
        }
    }
}
