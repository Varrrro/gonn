package layer

import (
	"github.com/varrrro/gonn/internal/functions"
	"gonum.org/v1/gonum/mat"
)

// ReluLayer with rectified linear activation function.
type ReluLayer struct {
	inputSize  int
	outputSize int
	input      mat.Vector
	output     mat.Vector
	weights    mat.Matrix
	biases     mat.Vector
	deltas     mat.Vector
	netInputs  mat.Vector
}

// CreateReluLayer with the given size.
func CreateReluLayer(nInput, nOutput int, weights, biases []float64) *ReluLayer {
	return &ReluLayer{
		inputSize:  nInput,
		outputSize: nOutput,
		weights:    mat.NewDense(nOutput, nInput, weights),
		biases:     mat.NewVecDense(nOutput, biases),
	}
}

// GetOutput of the layer.
func (l *ReluLayer) GetOutput() mat.Vector {
	return l.output
}

// GetWeights of the layer.
func (l *ReluLayer) GetWeights() mat.Matrix {
	return l.weights
}

// GetDeltas of the layer.
func (l *ReluLayer) GetDeltas() mat.Vector {
	return l.deltas
}

// FeedForward an input through the layer.
func (l *ReluLayer) FeedForward(x mat.Vector) {
	l.input = x

	z := mat.NewVecDense(l.outputSize, nil)
	z.MulVec(l.weights, l.input)
	z.AddVec(z, l.biases)

	l.netInputs = z

	y := mat.NewVecDense(l.outputSize, nil)
	for i := 0; i < l.outputSize; i++ {
		value := functions.Relu(z.AtVec(i))
		y.SetVec(i, value)
	}

	l.output = y
}

// CalculateDeltas for the layer with the given target.
//
// Not implemented as this is a hidden layer.
func (l *ReluLayer) CalculateDeltas(t mat.Vector) {}

// CalculateHiddenDeltas for the layer with the values from the next layer.
func (l *ReluLayer) CalculateHiddenDeltas(nextDeltas mat.Vector, nextWeights mat.Matrix) {
	d := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		sum := 0.0
		for j := 0; j < nextDeltas.Len(); j++ {
			sum += nextDeltas.AtVec(j) * nextWeights.At(j, i)
		}

		value := 0.0
		if l.netInputs.AtVec(i) > 0.0 {
			value = sum
		}

		d.SetVec(i, value)
	}

	l.deltas = d
}

// UpdateWeights and biases of the layer with the given Eta.
func (l *ReluLayer) UpdateWeights(eta float64) {
	newWeights := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiases := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightIncrement := -1.0 * eta * l.deltas.AtVec(i) * l.input.AtVec(j)
			newWeights.Set(i, j, l.weights.At(i, j)+weightIncrement)
		}

		biasIncrement := -1.0 * eta * l.deltas.AtVec(i)
		newBiases.SetVec(i, l.biases.AtVec(i)+biasIncrement)
	}

	l.weights = newWeights
	l.biases = newBiases
}
