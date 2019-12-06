package main

import (
	"log"

	"github.com/varrrro/gonn/internal/layer"
	"github.com/varrrro/gonn/internal/model"
	"github.com/varrrro/gonn/internal/util"
	"gonum.org/v1/gonum/mat"
)

const (
	mnistURL    = "http://yann.lecun.com/exdb/mnist/"
	localPath   = "data/mnist/"
	trainImages = "train-images-idx3-ubyte.gz"
	trainLabels = "train-labels-idx1-ubyte.gz"
	testImages  = "t10k-images-idx3-ubyte.gz"
	testLabels  = "t10k-labels-idx1-ubyte.gz"
)

func main() {
	/*
		err := util.DownloadMNIST(mnistURL, localPath, trainImages, trainLabels, testImages, testLabels)
		if err != nil {
			log.Println(err.Error())
			return
		}
	*/

	rawTrainImgs, trainLbls, err := util.ReadData(localPath, trainImages, trainLabels)
	if err != nil {
		log.Println(err.Error())
		return
	}

	rawTestImgs, testLbls, err := util.ReadData(localPath, testImages, testLabels)
	if err != nil {
		log.Println(err.Error())
		return
	}

	trainImgs := util.NormalizeImages(&rawTrainImgs)
	testImgs := util.NormalizeImages(&rawTestImgs)

	initNetwork(&trainImgs, &testImgs, &trainLbls, &testLbls)
}

func initNetwork(trainImgs, testImgs *[]mat.Vector, trainLabels, testLabels *[]int) {
	// Initialize weight and bias arrays
	hiddenWeights := util.InitializeRandom(784 * 256)
	hiddenBiases := util.InitializeRandom(256)
	outputWeights := util.InitializeRandom(256 * 10)
	outputBiases := util.InitializeRandom(10)

	// Create the layers
	hiddenLayer := layer.CreateSigmoidalLayer(784, 256, hiddenWeights, hiddenBiases)
	outputLayer := layer.CreateSoftmaxLayer(256, 10, outputWeights, outputBiases)

	// Create the nerwork parameters
	params := model.CreateParameters(0.1)

	// Create the network
	nn := model.Network{Layers: []model.Layer{hiddenLayer, outputLayer}, Params: params}

	// Train the network with the training set for a number of epochs
	nn.Train(*trainImgs, *trainLabels, 10)

	// Test the network with the test set
	nn.Test(*testImgs, *testLabels)
}
