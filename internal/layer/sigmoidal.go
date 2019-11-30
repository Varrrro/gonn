package layer

import (
	"gonum.org/v1/gonum/mat"

	"github.com/varrrro/gonn/internal/activation"
	"github.com/varrrro/gonn/internal/util"
)

// SigmoidalLayer with logistic activation function.
type SigmoidalLayer struct {
	InputSize    int
	OutputSize   int
	Input        mat.Vector
	Output       mat.Vector
	Weights      mat.Matrix
	WeightDeltas mat.Matrix
	NeuronDeltas mat.Vector
}

// GetOutput of this layer.
func (l *SigmoidalLayer) GetOutput() mat.Vector {
	return l.Output
}

// CreateSigmoidalLayer with the given values.
func CreateSigmoidalLayer(nInput, nOutput int, weights *[]float64) *SigmoidalLayer {
	return &SigmoidalLayer{
		InputSize:    nInput,
		OutputSize:   nOutput,
		Weights:      mat.NewDense(nOutput, nInput, *weights),
		WeightDeltas: mat.NewDense(nOutput, nInput, util.InitializeWeightDeltas(nInput, nOutput)),
	}
}

// FeedForward a set of features through this layer.
func (l *SigmoidalLayer) FeedForward(features mat.Vector) {
	l.Input = features

	output := mat.NewVecDense(l.OutputSize, nil)
	z := mat.NewVecDense(l.OutputSize, nil)
	z.MulVec(l.Weights, features)

	for i := 0; i < l.OutputSize; i++ {
		output.SetVec(i, activation.Logistic(z.AtVec(i)))
	}

	l.Output = output
}

// CalculateNeuronDeltas with the error gradient.
func (l *SigmoidalLayer) CalculateNeuronDeltas(gradient mat.Vector) {
	deltas := mat.NewVecDense(l.OutputSize, nil)

	for i := 0; i < l.OutputSize; i++ {
		value := l.Output.AtVec(i) * (1 - l.Output.AtVec(i)) * gradient.AtVec(i)
		deltas.SetVec(i, value)
	}

	l.NeuronDeltas = deltas
}

// CalculateGradient of the error in this layer based on the next.
func (l *SigmoidalLayer) CalculateGradient(deltas mat.Vector, weights mat.Matrix) mat.Vector {
	gradient := mat.NewVecDense(l.OutputSize, nil)
	n, _ := weights.Dims()

	for i := 0; i < l.OutputSize; i++ {
		sum := 0.0

		for j := 0; j < n; j++ {
			sum += weights.At(j, i) * deltas.AtVec(j)
		}

		gradient.SetVec(i, sum)
	}

	return gradient
}

// DoMomentumStep with the given Mu.
func (l *SigmoidalLayer) DoMomentumStep(mu float64) {
	increment := mat.NewDense(l.OutputSize, l.InputSize, nil)
	newWeights := mat.NewDense(l.OutputSize, l.InputSize, nil)

	increment.Scale(mu, l.WeightDeltas)
	newWeights.Add(l.Weights, increment)

	l.Weights = newWeights
}

// DoCorrectionStep with the given Eta.
func (l *SigmoidalLayer) DoCorrectionStep(eta float64) {
	deltas := mat.NewDense(l.OutputSize, l.InputSize, nil)
	newWeights := mat.NewDense(l.OutputSize, l.InputSize, nil)
	newWeightDeltas := mat.NewDense(l.OutputSize, l.InputSize, nil)

	for i := 0; i < l.OutputSize; i++ {
		for j := 0; j < l.InputSize; j++ {
			value := l.Input.AtVec(j) * l.NeuronDeltas.AtVec(i) * eta * -1

			deltas.Set(i, j, value)
		}
	}

	newWeights.Add(l.Weights, deltas)
	newWeightDeltas.Add(l.Weights, deltas)

	l.Weights = newWeights
	l.WeightDeltas = newWeightDeltas
}
