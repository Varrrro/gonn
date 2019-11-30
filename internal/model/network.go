package model

// Network composed of layers that extract features from an input.
type Network struct {
	Layers []Layer
	Params Parameters
}
