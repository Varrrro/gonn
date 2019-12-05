package layer

import (
	"github.com/varrrro/gonn/internal/functions"
	"github.com/varrrro/gonn/internal/util"
	"gonum.org/v1/gonum/mat"
)

// SigmoidalLayer with logistic activation function.
type SigmoidalLayer struct {
	inputSize        int
	outputSize       int
	input            mat.Vector
	output           mat.Vector
	weights          mat.Matrix
	biases           mat.Vector
	deltas           mat.Vector
	weightIncrements mat.Matrix
	biasIncrements   mat.Vector
}

// CreateSigmoidalLayer with the given size.
func CreateSigmoidalLayer(nInput, nOutput int) *SigmoidalLayer {
	return &SigmoidalLayer{
		inputSize:        nInput,
		outputSize:       nOutput,
		weights:          mat.NewDense(nOutput, nInput, util.InitializeRandom(nInput*nOutput)),
		biases:           mat.NewVecDense(nOutput, util.InitializeRandom(nOutput)),
		weightIncrements: mat.NewDense(nOutput, nInput, util.InitializeZeroes(nInput*nOutput)),
		biasIncrements:   mat.NewVecDense(nOutput, util.InitializeZeroes(nOutput)),
	}
}

// GetOutput of the layer.
func (l *SigmoidalLayer) GetOutput() mat.Vector {
	return l.output
}

// GetWeights of the layer.
func (l *SigmoidalLayer) GetWeights() mat.Matrix {
	return l.weights
}

// GetDeltas of the layer.
func (l *SigmoidalLayer) GetDeltas() mat.Vector {
	return l.deltas
}

// FeedForward an input through the layer.
func (l *SigmoidalLayer) FeedForward(x mat.Vector) {
	l.input = x

	z := mat.NewVecDense(l.outputSize, nil)
	z.MulVec(l.weights, l.input)
	z.AddVec(z, l.biases)

	y := mat.NewVecDense(l.outputSize, nil)
	for i := 0; i < l.outputSize; i++ {
		value := functions.Logistic(z.AtVec(i))
		y.SetVec(i, value)
	}

	l.output = y
}

// CalculateDeltas for the layer with the given target.
//
// Not implemented in a hidden layer.
func (l *SigmoidalLayer) CalculateDeltas(t mat.Vector) {
	diff := mat.NewVecDense(l.outputSize, nil)
	diff.SubVec(l.output, t)

	d := mat.NewVecDense(l.outputSize, nil)
	for i := 0; i < l.outputSize; i++ {
		value := l.output.AtVec(i) * (1.0 - l.output.AtVec(i)) * diff.AtVec(i)
		d.SetVec(i, value)
	}

	l.deltas = d
}

// CalculateHiddenDeltas for the layer with the values from the next layer.
func (l *SigmoidalLayer) CalculateHiddenDeltas(nextDeltas mat.Vector, nextWeights mat.Matrix) {
	d := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		sum := 0.0
		for j := 0; j < nextDeltas.Len(); j++ {
			sum += nextDeltas.AtVec(j) * nextWeights.At(j, i)
		}

		value := l.output.AtVec(i) * (1.0 - l.output.AtVec(i)) * sum
		d.SetVec(i, value)
	}

	l.deltas = d
}

// UpdateWeights and biases of the layer with the given Eta.
func (l *SigmoidalLayer) UpdateWeights(eta, mu float64) {
	newWeights := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiases := mat.NewVecDense(l.outputSize, nil)

	newWeightIncrements := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiasIncrements := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightIncrement := (mu * l.weightIncrements.At(i, j)) - (eta * l.deltas.AtVec(i) * l.input.AtVec(j))
			newWeights.Set(i, j, l.weights.At(i, j)+weightIncrement)
			newWeightIncrements.Set(i, j, weightIncrement)
		}

		biasIncrement := (mu * l.biasIncrements.AtVec(i)) - (eta * l.deltas.AtVec(i))
		newBiases.SetVec(i, l.biases.AtVec(i)+biasIncrement)
		newBiasIncrements.SetVec(i, biasIncrement)
	}

	l.weights = newWeights
	l.biases = newBiases

	l.weightIncrements = newWeightIncrements
	l.biasIncrements = newBiasIncrements
}

// DoMomentumStep with the given Mu.
func (l *SigmoidalLayer) DoMomentumStep(mu float64) {
	newWeights := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiases := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightIncrement := (mu * l.weightIncrements.At(i, j))
			newWeights.Set(i, j, l.weights.At(i, j)+weightIncrement)
		}

		biasIncrement := (mu * l.biasIncrements.AtVec(i))
		newBiases.SetVec(i, l.biases.AtVec(i)+biasIncrement)
	}

	l.weights = newWeights
	l.biases = newBiases
}

// DoCorrectionStep with the given Eta.
func (l *SigmoidalLayer) DoCorrectionStep(eta float64) {
	newWeights := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiases := mat.NewVecDense(l.outputSize, nil)

	newWeightIncrements := mat.NewDense(l.outputSize, l.inputSize, nil)
	newBiasIncrements := mat.NewVecDense(l.outputSize, nil)

	for i := 0; i < l.outputSize; i++ {
		for j := 0; j < l.inputSize; j++ {
			weightIncrement := -1.0 * eta * l.deltas.AtVec(i) * l.input.AtVec(j)
			newWeights.Set(i, j, l.weights.At(i, j)+weightIncrement)
			newWeightIncrements.Set(i, j, weightIncrement)
		}

		biasIncrement := - -1.0 * eta * l.deltas.AtVec(i)
		newBiases.SetVec(i, l.biases.AtVec(i)+biasIncrement)
		newBiasIncrements.SetVec(i, biasIncrement)
	}

	l.weights = newWeights
	l.biases = newBiases

	l.weightIncrements = newWeightIncrements
	l.biasIncrements = newBiasIncrements
}
