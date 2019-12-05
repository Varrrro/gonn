package model

import "gonum.org/v1/gonum/mat"

// A Layer of a neural network.
type Layer interface {
	GetOutput() mat.Vector
	GetWeights() mat.Matrix
	GetDeltas() mat.Vector
	FeedForward(mat.Vector)
	CalculateDeltas(mat.Vector)
	CalculateHiddenDeltas(mat.Vector, mat.Matrix)
	UpdateWeights(float64, float64)
	DoMomentumStep(float64)
	DoCorrectionStep(float64)
}
