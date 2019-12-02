package model

import "gonum.org/v1/gonum/mat"

// A Layer of a neural network.
type Layer interface {
	GetOutput() mat.Vector
	GetNeuronDeltas() mat.Vector
	GetWeights() mat.Matrix
	FeedForward(features mat.Vector)
	CalculateNeuronDeltas(gradient mat.Vector)
	CalculateGradient(deltas mat.Vector, weights mat.Matrix) mat.Vector
	DoMomentumStep(mu float64)
	DoCorrectionStep(eta float64)
}
