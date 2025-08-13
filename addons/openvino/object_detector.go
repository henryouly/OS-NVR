package openvino

import (
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"sort"

	"google.golang.org/grpc"

	framework "github.com/figroc/tensorflow-serving-client/v2/go/tensorflow/core/framework"
	apis "github.com/figroc/tensorflow-serving-client/v2/go/tensorflow_serving/apis"
)

// BoundingBox defines the structure for a detected object.
type BoundingBox struct {
	ClassId int
	Conf    float32
	Xmin    float32
	Ymin    float32
	Xmax    float32
	Ymax    float32
}

const (
	// Model-specific constants
	detectionOutputKey = "output0"
	inputWidth         = 320.0
	inputHeight        = 320.0
	numClasses         = 80

	// Post-processing thresholds
	minConfidence = 0.5
	iouThreshold  = 0.45

	// sizeOfHalf is the size of a float16 in bytes.
	sizeOfHalf = 2
)

type ObjectDetector struct {
	conn      *grpc.ClientConn
	client    apis.PredictionServiceClient
	modelName string
	inputName string
}

func NewObjectDetector(grpcHost, modelName, inputName string) (*ObjectDetector, error) {
	conn, err := grpc.Dial(grpcHost, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("could not connect to gRPC server: %w", err)
	}
	client := apis.NewPredictionServiceClient(conn)
	return &ObjectDetector{
		conn:      conn,
		client:    client,
		modelName: modelName,
		inputName: inputName,
	}, nil
}

func (od *ObjectDetector) Close() {
	if od.conn != nil {
		od.conn.Close()
	}
}

func (od *ObjectDetector) DetectObjects(img image.Image) ([]BoundingBox, error) {
	tensor, err := preprocessImage(img)
	if err != nil {
		return nil, fmt.Errorf("failed to preprocess image: %w", err)
	}

	request := &apis.PredictRequest{
		ModelSpec: &apis.ModelSpec{Name: od.modelName},
		Inputs: map[string]*framework.TensorProto{
			od.inputName: tensor,
		},
	}

	response, err := od.client.Predict(context.Background(), request)
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %w", err)
	}

	return decodeResponse(response)
}

// preprocessImage resizes, normalizes, and converts image data to a DT_HALF tensor.
func preprocessImage(img image.Image) (*framework.TensorProto, error) {
	numPixels := int(inputWidth * inputHeight)

	// 2. Pre-allocate a single slice for the final NCHW data.
	// This avoids creating three intermediate slices and concatenating them.
	halfVals := make([]int32, 3*numPixels)

	// 3. Iterate, convert, and place data directly into the final slice.
	for y := 0; y < int(inputHeight); y++ {
		for x := 0; x < int(inputWidth); x++ {
			pixelIdx := y*int(inputWidth) + x
			c := img.At(x, y)
			r, g, b, _ := c.RGBA() // Ignore alpha channel

			// Directly normalize, convert to half-float, and place into the correct
			// channel plane within the single `halfVals` slice.
			// NCHW format: R-plane, then G-plane, then B-plane.
			
			// Red channel -> first plane
			halfVals[pixelIdx] = int32(float32ToHalf(float32(r) / 255.0))
			// Green channel -> second plane (offset by the size of one plane)
			halfVals[pixelIdx+numPixels] = int32(float32ToHalf(float32(g) / 255.0))
			// Blue channel -> third plane (offset by the size of two planes)
			halfVals[pixelIdx+(2*numPixels)] = int32(float32ToHalf(float32(b) / 255.0))
		}
	}

	// 4. Create the tensor
	return &framework.TensorProto{
		Dtype: framework.DataType_DT_HALF,
		TensorShape: &framework.TensorShapeProto{
			Dim: []*framework.TensorShapeProto_Dim{
				{Size: 1},
				{Size: 3},
				{Size: inputHeight},
				{Size: inputWidth},
			},
		},
		HalfVal: halfVals,
	}, nil
}

// decodeResponse interprets the model's raw DT_HALF output tensor.
func decodeResponse(response *apis.PredictResponse) ([]BoundingBox, error) {
	output, ok := response.Outputs[detectionOutputKey]
	if !ok {
		return nil, fmt.Errorf("output key '%s' not found in response", detectionOutputKey)
	}

	if output.Dtype != framework.DataType_DT_HALF {
		return nil, fmt.Errorf("unexpected output dtype: got %v, want %v", output.Dtype, framework.DataType_DT_HALF)
	}

	dims := output.TensorShape.Dim
	if len(dims) != 3 {
		return nil, fmt.Errorf("expected 3 dimensions in output, got %d", len(dims))
	}
	numProposals := int(dims[2].Size)
	outputSize := int(dims[1].Size)
	if outputSize != 4+numClasses {
		return nil, fmt.Errorf("output size mismatch: expected %d, got %d", 4+numClasses, outputSize)
	}

	// Decode half-precision float tensor content into a standard float32 slice.
	outputData, err := decodeHalfTensor(output)
	if err != nil {
		return nil, err
	}

	var candidates []BoundingBox
	for i := 0; i < numProposals; i++ {
		var maxClassScore float32 = 0.0
		var maxClassID int = -1

		for j := 0; j < numClasses; j++ {
			classScoreIndex := (4+j)*numProposals + i
			classScore := outputData[classScoreIndex]
			if classScore > maxClassScore {
				maxClassScore = classScore
				maxClassID = j
			}
		}

		if maxClassScore < minConfidence {
			continue
		}

		cx := outputData[0*numProposals+i]
		cy := outputData[1*numProposals+i]
		w := outputData[2*numProposals+i]
		h := outputData[3*numProposals+i]

		box := BoundingBox{
			ClassId: maxClassID,
			Conf:  maxClassScore,
			Xmin:  cx - w*0.5,
			Ymin:  cy - h*0.5,
			Xmax:  cx + w*0.5,
			Ymax:  cy + h*0.5,
		}
		candidates = append(candidates, box)
	}

	return performNMS(candidates, iouThreshold)
}

// performNMS filters bounding boxes based on their Intersection over Union (IoU).
func performNMS(boxes []BoundingBox, iouThresh float32) ([]BoundingBox, error) {
	if len(boxes) == 0 {
		return []BoundingBox{}, nil
	}
	sort.Slice(boxes, func(i, j int) bool { return boxes[i].Conf > boxes[j].Conf })

	var finalBoxes []BoundingBox
	for len(boxes) > 0 {
		bestBox := boxes[0]
		finalBoxes = append(finalBoxes, bestBox)
		remainingBoxes := []BoundingBox{}
		for i := 1; i < len(boxes); i++ {
			if iou := calculateIoU(bestBox, boxes[i]); iou < iouThresh {
				remainingBoxes = append(remainingBoxes, boxes[i])
			}
		}
		boxes = remainingBoxes
	}
	return finalBoxes, nil
}

// calculateIoU computes the Intersection over Union of two bounding boxes.
func calculateIoU(box1, box2 BoundingBox) float32 {
	x1 := float32(math.Max(float64(box1.Xmin), float64(box2.Xmin)))
	y1 := float32(math.Max(float64(box1.Ymin), float64(box2.Ymin)))
	x2 := float32(math.Min(float64(box1.Xmax), float64(box2.Xmax)))
	y2 := float32(math.Min(float64(box1.Ymax), float64(box2.Ymax)))

	intersectionArea := float32(math.Max(0, float64(x2-x1))) * float32(math.Max(0, float64(y2-y1)))
	if intersectionArea == 0 {
		return 0
	}
	box1Area := (box1.Xmax - box1.Xmin) * (box1.Ymax - box1.Ymin)
	box2Area := (box2.Xmax - box2.Xmin) * (box2.Ymax - box2.Ymin)

	return intersectionArea / (box1Area + box2Area - intersectionArea)
}

// --- Half-Float Conversion Helpers ---

// decodeHalfTensor reads a DT_HALF tensor and returns its data as a []float32.
func decodeHalfTensor(tensor *framework.TensorProto) ([]float32, error) {
	if len(tensor.TensorContent) > 0 {
		numHalfs := len(tensor.TensorContent) / sizeOfHalf
		floats := make([]float32, numHalfs)
		for i := 0; i < numHalfs; i++ {
			half := binary.LittleEndian.Uint16(tensor.TensorContent[i*sizeOfHalf:])
			floats[i] = halfToFloat32(half)
		}
		return floats, nil
	}

	if len(tensor.HalfVal) > 0 {
		floats := make([]float32, len(tensor.HalfVal))
		for i, halfInt := range tensor.HalfVal {
			floats[i] = halfToFloat32(uint16(halfInt))
		}
		return floats, nil
	}

	return nil, fmt.Errorf("no data found in half-float tensor")
}

// float32ToHalf converts a 32-bit float to a 16-bit half-precision float.
func float32ToHalf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff) - 127
	mant := bits & 0x7fffff

	if exp > 15 { // Exponent overflow -> infinity
		return sign | 0x7c00
	}
	if exp < -14 { // Exponent underflow -> flush to zero
		return sign
	}
	exp += 15
	mant >>= 13
	return sign | uint16(exp<<10) | uint16(mant)
}

// halfToFloat32 converts a 16-bit half-precision float (as uint16) to a 32-bit float.
func halfToFloat32(h uint16) float32 {
	sign32 := uint32(h&0x8000) << 16
	exp16 := (h & 0x7C00) >> 10
	mant16 := uint32(h & 0x03FF)

	var bits32 uint32

	if exp16 == 0x1F { // Infinity or NaN
		bits32 = sign32 | 0x7F800000 | (mant16 << 13)
	} else if exp16 == 0 { // Zero or subnormal
		if mant16 != 0 {
			// Subnormal number. Normalize it.
			exp32 := uint32(127 - 14) // Start from the minimum exponent for normals
			for (mant16 & 0x0400) == 0 {
				mant16 <<= 1
				exp32--
			}
			mant32 := (mant16 & 0x03FF) << 13
			bits32 = sign32 | (exp32 << 23) | mant32
		} else { // Zero
			bits32 = sign32
		}
	} else { // Normal number
		exp32 := uint32(exp16) - 15 + 127
		mant32 := mant16 << 13
		bits32 = sign32 | (exp32 << 23) | mant32
	}

	return math.Float32frombits(bits32)
}
