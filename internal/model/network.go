package model

import (
	"log"
	"time"

	"github.com/varrrro/gonn/internal/util"
	"gonum.org/v1/gonum/mat"
)

// Network composed of layers that extract features from an input.
type Network struct {
	Layers []Layer
	Params Parameters
}

// GetOutput of the network.
func (n *Network) GetOutput() mat.Vector {
	return n.Layers[len(n.Layers)-1].GetOutput()
}

// FeedForward a pattern through the network.
func (n *Network) FeedForward(pattern mat.Vector) {
	n.Layers[0].FeedForward(pattern)

	for i := 0; i < len(n.Layers)-1; i++ {
		output := n.Layers[i].GetOutput()
		n.Layers[i+1].FeedForward(output)
	}
}

// Backpropagate the error through the network, updating the weights.
func (n *Network) Backpropagate(target mat.Vector) {
	output := n.GetOutput()

	diff := mat.NewVecDense(output.Len(), nil)
	diff.SubVec(output, target)

	n.Layers[len(n.Layers)-1].CalculateNeuronDeltas(diff)

	for i := len(n.Layers) - 2; i >= 0; i++ {
		nextLayerDeltas := n.Layers[i+1].GetNeuronDeltas()
		nextLayerWeights := n.Layers[i+1].GetWeights()

		gradient := n.Layers[i].CalculateGradient(nextLayerDeltas, nextLayerWeights)
		n.Layers[i].CalculateNeuronDeltas(gradient)
	}
}

// DoMomentumStep at each layer.
func (n *Network) DoMomentumStep() {
	for _, l := range n.Layers {
		l.DoMomentumStep(n.Params.Mu)
	}
}

// DoCorrectionStep at each layer.
func (n *Network) DoCorrectionStep() {
	for _, l := range n.Layers {
		l.DoCorrectionStep(n.Params.Eta)
	}
}

// Train the network with the given patterns.
func (n *Network) Train(patterns []mat.Vector, labels []int, epochs int) {
	log.Println(">>> Training started")
	start := time.Now()

	for i := 0; i < epochs; i++ {
		for j, p := range patterns {
			target := util.InitializeTarget(10, labels[j])

			n.DoMomentumStep()
			n.FeedForward(p)
			n.Backpropagate(target)
			n.DoCorrectionStep()
		}
	}

	elapsed := time.Since(start)
	log.Printf(">>> Training finished | Time elapsed: %f", elapsed.Seconds())
}

// Test the network with the given patterns.
func (n *Network) Test(patterns []mat.Vector, labels []int) {
	failures := 0

	for i, p := range patterns {
		n.FeedForward(p)
		output := n.GetOutput()

		max := mat.Max(output)

		if max != output.AtVec(labels[i]) {
			failures++
		}
	}

	failureRate := (float64(failures) / float64(len(patterns))) * float64(100)

	log.Println(">>>> Test finished")
	log.Printf("Number of patterns: %d", len(patterns))
	log.Printf("Number of failures: %d", failures)
	log.Printf("Falure rate: %f", failureRate)
}
